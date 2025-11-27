#!/usr/bin/env python3
"""
draft_questions.py - Generate Yes/No questions for taxonomy nodes

Incremental question generation with progress tracking.
Uses Claude Haiku 4.5 for LLM-based generation.

Usage:
    python draft_questions.py \
        --taxonomy data/taxonomy.jsonl \
        --questions data/questions.jsonl \
        --traits rules/traits_core.yaml \
        --progress data/progress.json \
        --max 10 \
        --use-llm true
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import re

try:
    import yaml
except ImportError:
    print("Warning: PyYAML not installed. Install with: pip install pyyaml")
    yaml = None


# LLM Configuration
LLM_MODEL = "claude-haiku-4-5-20251015"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.7

# System prompt for generating taxonomic questions
LLM_SYSTEM_PROMPT = """あなたは分類学の専門家です。動物分類ゲーム用のYes/No質問を生成します。

## 絶対に守るべきルール

1. **形態・解剖学的特徴のみ使用**
   - OK: 羽毛、乳腺、外骨格、歯の形状、体の構造、体色パターン
   - NG: 生息地、地理的分布、行動、食性、生態

2. **Yes/Noで明確に回答できる質問**
   - 曖昧さのない表現を使う
   - 「通常」「多くの場合」は避ける

3. **分類学的に正確**
   - その形質が対象グループを確実に区別できること

4. **日本語で出力**

## 出力形式（厳守）

```json
{
  "question": "質問文",
  "yes_taxa": ["該当するtaxonID1", "taxonID2"],
  "no_taxa": ["該当しないtaxonID1", "taxonID2"],
  "trait_used": "使用した形質（例：羽毛の有無）",
  "confidence": 0.0-1.0
}
```"""


class AnthropicClient:
    """Simple client for Anthropic API."""

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_message(self, model: str, max_tokens: int, system: str,
                       messages: List[dict], temperature: float = 0.7) -> Optional[dict]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "temperature": temperature
        }
        try:
            data = json.dumps(payload).encode('utf-8')
            request = Request(self.API_URL, data=data, headers=headers, method='POST')
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode('utf-8'))
        except HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ""
            print(f"  API Error {e.code}: {error_body[:200]}")
            return None
        except Exception as e:
            print(f"  Request error: {e}")
            return None

    def get_text_response(self, response: dict) -> Optional[str]:
        if not response:
            return None
        for block in response.get('content', []):
            if block.get('type') == 'text':
                return block.get('text', '')
        return None


class IncrementalDrafter:
    """Incremental question drafter with progress tracking."""

    def __init__(self, llm_client: Optional[AnthropicClient] = None):
        self.taxonomy: Dict[str, dict] = {}
        self.children_map: Dict[str, List[str]] = defaultdict(list)
        self.questions: List[dict] = []
        self.question_ids: Set[str] = set()
        self.questions_by_node: Dict[str, List[dict]] = defaultdict(list)
        self.traits: dict = {}
        self.progress: dict = {}
        self.llm_client = llm_client

    def load_taxonomy(self, path: Path) -> bool:
        """Load taxonomy (lightweight - just IDs and structure)."""
        if not path.exists():
            return False
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    node = json.loads(line)
                    tid = node.get('taxonID')
                    if tid:
                        self.taxonomy[tid] = node
                        parent = node.get('parentID')
                        if parent:
                            self.children_map[parent].append(tid)
        print(f"Loaded {len(self.taxonomy)} taxa")
        return True

    def load_questions(self, path: Path) -> bool:
        """Load existing questions."""
        if not path.exists():
            return True
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    self.questions.append(q)
                    qid = q.get('q_id')
                    if qid:
                        self.question_ids.add(qid)
                    nid = q.get('node_id')
                    if nid:
                        self.questions_by_node[nid].append(q)
        print(f"Loaded {len(self.questions)} existing questions")
        return True

    def load_traits(self, path: Path) -> bool:
        """Load trait definitions."""
        if not yaml or not path.exists():
            return False
        with open(path, 'r', encoding='utf-8') as f:
            self.traits = yaml.safe_load(f) or {}
        print(f"Loaded {len(self.traits)} trait entries")
        return True

    def load_progress(self, path: Path) -> bool:
        """Load progress tracking file."""
        if not path.exists():
            self.progress = {
                "completed_nodes": [],
                "last_processed_index": 0,
                "stats": {"total_questions": 0, "llm_generated": 0}
            }
            return True
        with open(path, 'r', encoding='utf-8') as f:
            self.progress = json.load(f)
        print(f"Progress: {len(self.progress.get('completed_nodes', []))} nodes completed")
        return True

    def save_progress(self, path: Path):
        """Save progress tracking file."""
        self.progress["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.progress["stats"]["total_questions"] = len(self.questions)
        self.progress["stats"]["nodes_with_questions"] = len(self.questions_by_node)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)

    def save_questions(self, path: Path):
        """Save all questions."""
        # Sort by node_id, then q_id
        self.questions.sort(key=lambda q: (q.get('node_id', ''), q.get('q_id', '')))
        with open(path, 'w', encoding='utf-8') as f:
            for q in self.questions:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')

    def get_pending_nodes(self) -> List[str]:
        """Get nodes that need more questions, sorted by priority."""
        completed = set(self.progress.get("completed_nodes", []))
        pending = []

        for node_id, node in self.taxonomy.items():
            if node_id in completed:
                continue
            children = self.children_map.get(node_id, [])
            if len(children) < 2:
                # Leaf or single child - mark as complete
                completed.add(node_id)
                continue

            # Calculate how many questions we need
            import math
            needed = max(1, int(math.ceil(math.log2(len(children)))))
            existing = len(self.questions_by_node.get(node_id, []))

            if existing < needed:
                priority = len(children)  # More children = higher priority
                pending.append((node_id, priority, needed - existing))

        # Sort by priority (most children first)
        pending.sort(key=lambda x: -x[1])
        return [p[0] for p in pending]

    def get_node_context(self, node_id: str) -> dict:
        """Get context about a node for LLM prompt (minimal data)."""
        node = self.taxonomy.get(node_id, {})
        children = self.children_map.get(node_id, [])

        # Get trait info for this node and children
        node_trait = self.traits.get(node_id, {})
        children_info = []

        for cid in children[:15]:  # Limit to avoid huge prompts
            child = self.taxonomy.get(cid, {})
            child_trait = self.traits.get(cid, {})
            info = {
                "id": cid,
                "name": child.get('vernacularName') or child.get('scientificName', cid),
                "rank": child.get('rank', ''),
            }
            if isinstance(child_trait, dict) and 'hint' in child_trait:
                info["trait"] = child_trait['hint']
            children_info.append(info)

        return {
            "node_id": node_id,
            "name": node.get('vernacularName') or node.get('scientificName', node_id),
            "rank": node.get('rank', ''),
            "parent_trait": node_trait.get('hint') if isinstance(node_trait, dict) else None,
            "children": children_info,
            "existing_questions": [q['question'] for q in self.questions_by_node.get(node_id, [])]
        }

    def generate_llm_question(self, node_id: str) -> Optional[dict]:
        """Generate a question using LLM."""
        if not self.llm_client:
            return None

        ctx = self.get_node_context(node_id)
        if len(ctx["children"]) < 2:
            return None

        # Build compact prompt
        children_text = "\n".join([
            f"- {c['name']} ({c['id']})" + (f" [形質: {c['trait']}]" if c.get('trait') else "")
            for c in ctx["children"]
        ])

        existing_qs = "\n".join([f"- {q}" for q in ctx["existing_questions"]]) if ctx["existing_questions"] else "なし"

        prompt = f"""対象: {ctx['name']} ({node_id}, {ctx['rank']})
{f"親の形質: {ctx['parent_trait']}" if ctx['parent_trait'] else ""}

子の一覧:
{children_text}

既存の質問（これと異なる形質を使う）:
{existing_qs}

上記の子を二分するYes/No質問を1つ生成。形態的特徴のみ使用。JSON形式で出力。"""

        print(f"  [LLM] {node_id}...")

        try:
            response = self.llm_client.create_message(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                system=LLM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE
            )
            text = self.llm_client.get_text_response(response)
            if not text:
                return None

            # Parse JSON
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if not json_match:
                return None

            result = json.loads(json_match.group())

            # Build question object
            yes_taxa = result.get('yes_taxa', [])
            no_taxa = result.get('no_taxa', [])

            q = {
                'node_id': node_id,
                'rank': ctx['rank'],
                'q_id': f"{node_id}_llm_{len(self.questions_by_node.get(node_id, []))}",
                'question': result.get('question', ''),
                'yes_next': '_'.join(yes_taxa[:3]) if yes_taxa else f"{node_id}_yes",
                'no_next': '_'.join(no_taxa[:3]) if no_taxa else f"{node_id}_no",
                'sources': ['llm', 'claude-haiku-4.5'],
                'confidence': float(result.get('confidence', 0.75)),
                'trait_used': result.get('trait_used', '')
            }

            if q['question'] and q['q_id'] not in self.question_ids:
                print(f"    -> {q['question'][:50]}...")
                return q

        except Exception as e:
            print(f"    Error: {e}")

        return None

    def generate_rule_based_question(self, node_id: str) -> Optional[dict]:
        """Generate question from traits_core.yaml rules."""
        ctx = self.get_node_context(node_id)
        children = ctx["children"]

        if len(children) < 2:
            return None

        # Find a child with a distinguishing trait
        for child in children:
            if child.get("trait"):
                q_id = f"{node_id}_rule_{len(self.questions_by_node.get(node_id, []))}"
                if q_id in self.question_ids:
                    continue

                return {
                    'node_id': node_id,
                    'rank': ctx['rank'],
                    'q_id': q_id,
                    'question': f"{child['trait']}がありますか？",
                    'yes_next': child['id'],
                    'no_next': f"{node_id}_minus_{child['id']}",
                    'sources': ['traits_core'],
                    'confidence': 0.85
                }

        return None

    def draft(self, max_questions: int, use_llm: bool = True) -> int:
        """Generate questions incrementally."""
        pending = self.get_pending_nodes()
        print(f"\n{len(pending)} nodes need questions")

        if not pending:
            print("All nodes have sufficient questions!")
            return 0

        drafted = 0
        llm_count = 0

        for node_id in pending:
            if drafted >= max_questions:
                break

            # Try LLM first
            q = None
            if use_llm and self.llm_client:
                q = self.generate_llm_question(node_id)
                if q:
                    llm_count += 1
                    time.sleep(0.3)  # Rate limiting

            # Fall back to rule-based
            if not q:
                q = self.generate_rule_based_question(node_id)

            if q and q['q_id'] not in self.question_ids:
                self.questions.append(q)
                self.question_ids.add(q['q_id'])
                self.questions_by_node[q['node_id']].append(q)
                drafted += 1

                # Check if node is now complete
                children = self.children_map.get(node_id, [])
                import math
                needed = max(1, int(math.ceil(math.log2(len(children))))) if children else 0
                if len(self.questions_by_node[node_id]) >= needed:
                    if "completed_nodes" not in self.progress:
                        self.progress["completed_nodes"] = []
                    self.progress["completed_nodes"].append(node_id)

        self.progress["stats"]["llm_generated"] = self.progress.get("stats", {}).get("llm_generated", 0) + llm_count
        print(f"\nDrafted {drafted} questions ({llm_count} via LLM)")
        return drafted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--traits", type=Path, required=True)
    parser.add_argument("--progress", type=Path, default=Path("data/progress.json"))
    parser.add_argument("--max", type=int, default=10)
    parser.add_argument("--use-llm", choices=['true', 'false'], default='true')
    parser.add_argument("--out", type=Path)

    args = parser.parse_args()
    out_path = args.out or args.questions

    # Initialize LLM
    llm_client = None
    use_llm = args.use_llm == 'true'
    if use_llm:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if api_key:
            print(f"LLM: {LLM_MODEL}")
            llm_client = AnthropicClient(api_key)
        else:
            print("Warning: ANTHROPIC_API_KEY not set, using rule-based only")
            use_llm = False

    drafter = IncrementalDrafter(llm_client=llm_client)

    # Load data
    drafter.load_taxonomy(args.taxonomy)
    drafter.load_questions(args.questions)
    drafter.load_traits(args.traits)
    drafter.load_progress(args.progress)

    # Generate
    drafted = drafter.draft(args.max, use_llm=use_llm)

    # Save
    if drafted > 0:
        drafter.save_questions(out_path)
        drafter.save_progress(args.progress)
        print(f"Saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
