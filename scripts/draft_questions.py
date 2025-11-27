#!/usr/bin/env python3
"""
draft_questions.py - Generate Yes/No questions for taxonomy nodes

Reads taxonomy.jsonl and traits_core.yaml to generate questions that
split taxa into Yes/No groups based on taxonomic traits.

Supports LLM-based question generation using Claude Haiku 4.5.

Usage:
    python draft_questions.py \
        --taxonomy data/taxonomy.jsonl \
        --questions data/questions.jsonl \
        --rules rules/templates.yaml \
        --traits rules/traits_core.yaml \
        --max 40 \
        --out data/questions.jsonl

    # With LLM:
    python draft_questions.py \
        --taxonomy data/taxonomy.jsonl \
        --questions data/questions.jsonl \
        --traits rules/traits_core.yaml \
        --max 40 \
        --use-llm true \
        --out data/questions.jsonl
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from urllib.request import urlopen, Request
from urllib.parse import quote
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
   - OK: 羽毛、乳腺、外骨格、歯の形状、体の構造
   - NG: 生息地、地理的分布、行動、食性、生態

2. **Yes/Noで明確に回答できる質問**
   - 曖昧さのない表現を使う
   - 「通常」「多くの場合」は避ける

3. **分類学的に正確**
   - その形質が対象グループを確実に区別できること
   - 例外がある場合は質問に含めない

4. **日本語で出力**
   - 専門用語は使ってよいが、わかりやすく

## 出力形式

JSON形式で出力してください：
```json
{
  "question": "質問文",
  "yes_group": "Yesの場合に該当するグループ/分類群",
  "no_group": "Noの場合に該当するグループ/分類群",
  "trait_type": "形質の種類（形態/解剖/発生など）",
  "confidence": 0.0-1.0の確信度,
  "reasoning": "この質問を選んだ理由（短く）"
}
```"""


class AnthropicClient:
    """Simple client for Anthropic API using urllib (no dependencies)."""

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_message(
        self,
        model: str,
        max_tokens: int,
        system: str,
        messages: List[dict],
        temperature: float = 0.7
    ) -> Optional[dict]:
        """Send a message to the Anthropic API."""
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
                result = json.loads(response.read().decode('utf-8'))
                return result

        except HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else ""
            print(f"  API Error {e.code}: {error_body[:200]}")
            return None
        except Exception as e:
            print(f"  Request error: {e}")
            return None

    def get_text_response(self, response: dict) -> Optional[str]:
        """Extract text from API response."""
        if not response:
            return None
        content = response.get('content', [])
        for block in content:
            if block.get('type') == 'text':
                return block.get('text', '')
        return None


class QuestionDrafter:
    """Generates Yes/No questions for taxonomy nodes."""

    def __init__(self, llm_client: Optional[AnthropicClient] = None):
        self.taxonomy: Dict[str, dict] = {}  # taxonID -> node
        self.children_map: Dict[str, List[str]] = defaultdict(list)
        self.existing_questions: Dict[str, List[dict]] = defaultdict(list)  # node_id -> questions
        self.templates: dict = {}
        self.traits: dict = {}
        self.new_questions: List[dict] = []
        self.llm_client = llm_client

    def load_taxonomy(self, path: Path) -> bool:
        """Load taxonomy from JSONL file."""
        print(f"Loading taxonomy from {path}...")

        if not path.exists():
            print(f"Warning: Taxonomy file not found: {path}")
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    node = json.loads(line)
                    taxon_id = node.get('taxonID')
                    if taxon_id:
                        self.taxonomy[taxon_id] = node

            # Build children map
            for taxon_id, node in self.taxonomy.items():
                parent_id = node.get('parentID')
                if parent_id:
                    self.children_map[parent_id].append(taxon_id)

            print(f"Loaded {len(self.taxonomy)} taxa")
            return len(self.taxonomy) > 0

        except Exception as e:
            print(f"Error loading taxonomy: {e}")
            return False

    def load_questions(self, path: Path) -> bool:
        """Load existing questions from JSONL file."""
        print(f"Loading existing questions from {path}...")

        if not path.exists():
            print("No existing questions file - starting fresh")
            return True

        try:
            with open(path, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    q = json.loads(line)
                    node_id = q.get('node_id')
                    if node_id:
                        self.existing_questions[node_id].append(q)
                        count += 1

            print(f"Loaded {count} existing questions")
            return True

        except Exception as e:
            print(f"Error loading questions: {e}")
            return False

    def load_templates(self, path: Path) -> bool:
        """Load question templates from YAML file."""
        print(f"Loading templates from {path}...")

        if not yaml:
            print("Warning: PyYAML not available, using default templates")
            self.templates = {
                'default': {'question_template': '{hint}がありますか？'}
            }
            return True

        if not path.exists():
            print(f"Warning: Templates file not found: {path}")
            self.templates = {
                'default': {'question_template': '{hint}がありますか？'}
            }
            return True

        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.templates = yaml.safe_load(f) or {}
            print(f"Loaded {len(self.templates)} template entries")
            return True

        except Exception as e:
            print(f"Error loading templates: {e}")
            return False

    def load_traits(self, path: Path) -> bool:
        """Load trait definitions from YAML file."""
        print(f"Loading traits from {path}...")

        if not yaml:
            print("Warning: PyYAML not available, cannot load traits")
            return False

        if not path.exists():
            print(f"Warning: Traits file not found: {path}")
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.traits = yaml.safe_load(f) or {}
            print(f"Loaded {len(self.traits)} trait entries")
            return True

        except Exception as e:
            print(f"Error loading traits: {e}")
            return False

    def get_node_name(self, node_id: str) -> str:
        """Get display name for a node."""
        if node_id in self.taxonomy:
            node = self.taxonomy[node_id]
            return node.get('vernacularName') or node.get('scientificName', node_id)
        return node_id

    def get_children(self, node_id: str) -> List[str]:
        """Get children of a node (from taxonomy or traits)."""
        # First check taxonomy
        if node_id in self.children_map:
            return self.children_map[node_id]

        # Check traits for parent reference
        children = []
        for trait_id, trait_data in self.traits.items():
            if isinstance(trait_data, dict) and trait_data.get('parent') == node_id:
                children.append(trait_id)

        return children

    def has_sufficient_questions(self, node_id: str) -> bool:
        """Check if a node has enough questions to fully split its children."""
        children = self.get_children(node_id)
        if len(children) <= 1:
            return True  # Leaf or single child - no questions needed

        existing = self.existing_questions.get(node_id, [])
        # Need roughly log2(n) questions to split n children
        import math
        needed = max(1, int(math.ceil(math.log2(len(children)))))
        return len(existing) >= needed

    def generate_trait_question(self, node_id: str, trait_data: dict) -> Optional[dict]:
        """Generate a question from trait data."""
        hint = trait_data.get('hint')
        if not hint:
            return None

        # Get template
        rank = trait_data.get('rank', 'default')
        template_data = self.templates.get(rank, self.templates.get('default', {}))
        template = template_data.get('question_template', '{hint}がありますか？')

        # Generate question text
        question_text = template.format(
            hint=hint,
            target=self.get_node_name(node_id),
            parent=self.get_node_name(trait_data.get('parent', ''))
        )

        return {
            'node_id': trait_data.get('parent', node_id),
            'rank': rank,
            'q_id': f"{node_id}_trait",
            'question': question_text,
            'yes_next': node_id,
            'no_next': f"{trait_data.get('parent', 'root')}_minus_{node_id}",
            'sources': ['traits_core'],
            'confidence': 0.9
        }

    def generate_predefined_questions(self, node_id: str) -> List[dict]:
        """Generate questions from predefined question lists in traits."""
        questions = []

        # Check for _questions key
        questions_key = f"{node_id}_questions"
        if questions_key in self.traits:
            q_list = self.traits[questions_key]
            if isinstance(q_list, list):
                for q_def in q_list:
                    q = {
                        'node_id': node_id,
                        'rank': self.taxonomy.get(node_id, {}).get('rank', 'unknown'),
                        'q_id': f"{node_id}_{q_def.get('id', 'Q')}",
                        'question': q_def.get('text', ''),
                        'yes_next': q_def.get('yes_subtree', ''),
                        'no_next': q_def.get('no_subtree', ''),
                        'sources': ['traits_core'] + q_def.get('refs', []),
                        'confidence': q_def.get('confidence', 0.8)
                    }
                    questions.append(q)

        return questions

    def generate_binary_split_question(self, node_id: str, children: List[str]) -> Optional[dict]:
        """Generate a question that splits children into two groups."""
        if len(children) < 2:
            return None

        # Sort children alphabetically and split in half
        sorted_children = sorted(children)
        mid = len(sorted_children) // 2
        yes_group = sorted_children[:mid]
        no_group = sorted_children[mid:]

        # Try to find a distinguishing trait for the yes_group
        yes_trait = None
        for child_id in yes_group:
            if child_id in self.traits:
                trait_data = self.traits[child_id]
                if isinstance(trait_data, dict) and 'hint' in trait_data:
                    yes_trait = trait_data['hint']
                    break

        if not yes_trait:
            # Use group names as trait
            yes_names = [self.get_node_name(c) for c in yes_group[:3]]
            yes_trait = f"{', '.join(yes_names)}などのグループ"

        # Create question
        return {
            'node_id': node_id,
            'rank': self.taxonomy.get(node_id, {}).get('rank', 'unknown'),
            'q_id': f"{node_id}_split_{len(self.existing_questions.get(node_id, []))}",
            'question': f"{yes_trait}に該当しますか？",
            'yes_next': '_'.join(yes_group) if len(yes_group) <= 3 else f"{node_id}_group_yes",
            'no_next': '_'.join(no_group) if len(no_group) <= 3 else f"{node_id}_group_no",
            'sources': ['binary_split'],
            'confidence': 0.6
        }

    def fetch_wikipedia_confidence(self, taxon_name: str, trait_hint: str) -> float:
        """
        Fetch Wikipedia summary and calculate confidence based on keyword match.

        Uses Wikimedia REST API (free, no authentication required).
        """
        base_confidence = 0.7

        try:
            # URL encode the taxon name
            encoded_name = quote(taxon_name)
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_name}"

            with urlopen(url, timeout=5) as response:
                if response.status != 200:
                    return base_confidence

                data = json.loads(response.read().decode('utf-8'))
                extract = data.get('extract', '').lower()

                if not extract:
                    return base_confidence

                # Check for trait keywords in the extract
                hint_words = re.findall(r'\w+', trait_hint.lower())
                matches = sum(1 for word in hint_words if word in extract and len(word) > 3)

                if matches >= 2:
                    return min(0.95, base_confidence + 0.2)
                elif matches >= 1:
                    return min(0.9, base_confidence + 0.1)

        except (URLError, json.JSONDecodeError, Exception):
            pass

        return base_confidence

    def generate_llm_question(self, node_id: str, children: List[str]) -> Optional[dict]:
        """
        Generate a question using Claude Haiku 4.5.

        Args:
            node_id: The taxonomy node to generate a question for
            children: List of child taxon IDs

        Returns:
            A question dict or None if generation failed
        """
        if not self.llm_client:
            return None

        if len(children) < 2:
            return None

        # Build context about the node and its children
        node = self.taxonomy.get(node_id, {})
        node_name = node.get('vernacularName') or node.get('scientificName', node_id)
        node_rank = node.get('rank', 'unknown')

        # Get info about children
        children_info = []
        for child_id in children[:20]:  # Limit to avoid too long prompts
            child = self.taxonomy.get(child_id, {})
            child_name = child.get('vernacularName') or child.get('scientificName', child_id)
            child_rank = child.get('rank', '')

            # Check if we have trait info
            trait_info = ""
            if child_id in self.traits:
                trait_data = self.traits[child_id]
                if isinstance(trait_data, dict) and 'hint' in trait_data:
                    trait_info = f" - 特徴: {trait_data['hint']}"

            children_info.append(f"- {child_name} ({child_id}, {child_rank}){trait_info}")

        children_text = "\n".join(children_info)

        # Build the prompt
        user_prompt = f"""以下の分類群の子を二分するYes/No質問を1つ生成してください。

## 対象ノード
- 名前: {node_name}
- 学名/ID: {node_id}
- ランク: {node_rank}

## 子の一覧（これらを二分する質問が必要）
{children_text}

## 要件
- 形態・解剖学的特徴のみを使用（生息地・行動・食性は禁止）
- 子のうち約半数がYes、残りがNoになるような質問
- 確実に区別できる形質を選ぶ

JSON形式で1つだけ出力してください。"""

        print(f"  [LLM] Generating question for {node_id}...")

        try:
            response = self.llm_client.create_message(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                system=LLM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=LLM_TEMPERATURE
            )

            text = self.llm_client.get_text_response(response)
            if not text:
                print(f"  [LLM] No response for {node_id}")
                return None

            # Parse JSON from response
            # Try to find JSON in the response
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if not json_match:
                print(f"  [LLM] Could not parse JSON from response")
                return None

            llm_result = json.loads(json_match.group())

            # Convert LLM output to our question format
            question = {
                'node_id': node_id,
                'rank': node_rank,
                'q_id': f"{node_id}_llm_{len(self.existing_questions.get(node_id, []))}",
                'question': llm_result.get('question', ''),
                'yes_next': llm_result.get('yes_group', f"{node_id}_yes"),
                'no_next': llm_result.get('no_group', f"{node_id}_no"),
                'sources': ['llm', 'claude-haiku-4.5'],
                'confidence': float(llm_result.get('confidence', 0.75)),
                'llm_reasoning': llm_result.get('reasoning', '')
            }

            if question['question']:
                print(f"  [LLM] Generated: {question['question'][:60]}...")
                return question

        except json.JSONDecodeError as e:
            print(f"  [LLM] JSON parse error: {e}")
        except Exception as e:
            print(f"  [LLM] Error: {e}")

        return None

    def draft_questions(self, max_questions: int, use_wikipedia: bool = False, use_llm: bool = False) -> int:
        """Generate questions for nodes that need them."""
        print(f"\nDrafting up to {max_questions} new questions...")

        drafted = 0
        existing_q_ids = set()

        # Collect existing question IDs
        for qs in self.existing_questions.values():
            for q in qs:
                existing_q_ids.add(q.get('q_id'))

        # First, add predefined questions from traits
        for node_id in self.traits:
            if drafted >= max_questions:
                break

            if node_id.endswith('_questions'):
                continue

            # Check for predefined questions
            questions = self.generate_predefined_questions(node_id)
            for q in questions:
                if drafted >= max_questions:
                    break
                if q['q_id'] not in existing_q_ids:
                    self.new_questions.append(q)
                    existing_q_ids.add(q['q_id'])
                    drafted += 1
                    print(f"  + {q['q_id']}: {q['question'][:50]}...")

        # Then, generate trait-based questions
        for node_id, trait_data in self.traits.items():
            if drafted >= max_questions:
                break

            if node_id.endswith('_questions'):
                continue

            if not isinstance(trait_data, dict):
                continue

            parent_id = trait_data.get('parent')
            if not parent_id:
                continue

            # Generate question for this trait
            q = self.generate_trait_question(node_id, trait_data)
            if q and q['q_id'] not in existing_q_ids:
                # Optionally verify with Wikipedia
                if use_wikipedia and trait_data.get('hint'):
                    q['confidence'] = self.fetch_wikipedia_confidence(
                        node_id, trait_data['hint']
                    )
                    q['sources'].append('wikipedia')

                self.new_questions.append(q)
                existing_q_ids.add(q['q_id'])
                drafted += 1
                print(f"  + {q['q_id']}: {q['question'][:50]}...")

        # Finally, generate binary split questions for nodes with many children
        nodes_needing_questions = []
        for node_id in list(self.taxonomy.keys()) + list(self.traits.keys()):
            if node_id.endswith('_questions'):
                continue
            if not self.has_sufficient_questions(node_id):
                children = self.get_children(node_id)
                if len(children) >= 2:
                    nodes_needing_questions.append((node_id, len(children)))

        # Sort by number of children (prioritize nodes with more children)
        nodes_needing_questions.sort(key=lambda x: -x[1])

        for node_id, _ in nodes_needing_questions:
            if drafted >= max_questions:
                break

            children = self.get_children(node_id)

            # Try LLM first if enabled
            q = None
            if use_llm and self.llm_client:
                q = self.generate_llm_question(node_id, children)
                # Rate limiting: wait a bit between API calls
                if q:
                    time.sleep(0.5)

            # Fall back to binary split if LLM didn't work
            if not q:
                q = self.generate_binary_split_question(node_id, children)

            if q and q['q_id'] not in existing_q_ids:
                self.new_questions.append(q)
                existing_q_ids.add(q['q_id'])
                drafted += 1
                print(f"  + {q['q_id']}: {q['question'][:50]}...")

        print(f"\nDrafted {drafted} new questions")
        return drafted

    def write_questions(self, out_path: Path):
        """Write all questions (existing + new) to JSONL file."""
        print(f"\nWriting questions to {out_path}...")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge existing and new questions
        all_questions = []

        for qs in self.existing_questions.values():
            all_questions.extend(qs)

        all_questions.extend(self.new_questions)

        # Sort by node_id, then q_id
        all_questions.sort(key=lambda q: (q.get('node_id', ''), q.get('q_id', '')))

        # Remove duplicates
        seen_ids = set()
        unique_questions = []
        for q in all_questions:
            q_id = q.get('q_id')
            if q_id and q_id not in seen_ids:
                seen_ids.add(q_id)
                unique_questions.append(q)

        with open(out_path, 'w', encoding='utf-8') as f:
            for q in unique_questions:
                json_line = json.dumps(q, ensure_ascii=False)
                f.write(json_line + '\n')

        print(f"Wrote {len(unique_questions)} questions")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Yes/No questions for taxonomy nodes"
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        required=True,
        help="Path to taxonomy.jsonl"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to existing questions.jsonl"
    )
    parser.add_argument(
        "--rules",
        type=Path,
        help="Path to templates.yaml"
    )
    parser.add_argument(
        "--traits",
        type=Path,
        required=True,
        help="Path to traits_core.yaml"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=40,
        help="Maximum number of questions to generate"
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for questions.jsonl"
    )
    parser.add_argument(
        "--use-wikipedia",
        choices=['true', 'false'],
        default='false',
        help="Use Wikipedia API to verify traits"
    )
    parser.add_argument(
        "--use-llm",
        choices=['true', 'false'],
        default='false',
        help="Use Claude Haiku 4.5 to generate questions (requires ANTHROPIC_API_KEY)"
    )

    args = parser.parse_args()

    # Initialize LLM client if requested
    llm_client = None
    use_llm = args.use_llm == 'true'

    if use_llm:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            print("Warning: --use-llm=true but ANTHROPIC_API_KEY not set")
            print("  Set it with: export ANTHROPIC_API_KEY='your-key'")
            print("  Falling back to rule-based generation")
            use_llm = False
        else:
            print(f"LLM enabled: using {LLM_MODEL}")
            llm_client = AnthropicClient(api_key)

    drafter = QuestionDrafter(llm_client=llm_client)

    # Load data
    drafter.load_taxonomy(args.taxonomy)
    drafter.load_questions(args.questions)

    if args.rules:
        drafter.load_templates(args.rules)
    else:
        drafter.templates = {'default': {'question_template': '{hint}がありますか？'}}

    if not drafter.load_traits(args.traits):
        print("Error: Failed to load traits file")
        return 1

    # Generate questions
    use_wikipedia = args.use_wikipedia == 'true'
    drafted = drafter.draft_questions(args.max, use_wikipedia=use_wikipedia, use_llm=use_llm)

    # Write output
    drafter.write_questions(args.out)

    print(f"\nQuestion drafting complete! Generated {drafted} new questions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
