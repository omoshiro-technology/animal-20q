/**
 * Animal 20Q: Taxonomy Edition
 * 分類学的特徴で動物を当てるYes/Noゲーム
 */

class Animal20Q {
  constructor() {
    // Data
    this.taxonomy = new Map();      // taxonID -> node
    this.questions = new Map();     // node_id -> questions[]
    this.childrenMap = new Map();   // taxonID -> children[]

    // Game state
    this.state = null;
    this.questionCount = 0;
    this.history = [];

    // DOM elements
    this.screens = {
      loading: document.getElementById('loading'),
      start: document.getElementById('start-screen'),
      game: document.getElementById('game-screen'),
      result: document.getElementById('result-screen'),
      error: document.getElementById('error-screen'),
    };

    this.elements = {
      breadcrumb: document.getElementById('breadcrumb'),
      questionNum: document.getElementById('q-num'),
      questionText: document.getElementById('question-text'),
      confidence: document.getElementById('confidence'),
      sources: document.getElementById('sources'),
      candidatesCount: document.getElementById('candidates-count'),
      resultCard: document.getElementById('result-card'),
      resultTitle: document.getElementById('result-title'),
      resultContent: document.getElementById('result-content'),
      errorMessage: document.getElementById('error-message'),
      creditsModal: document.getElementById('credits-modal'),
    };

    this.bindEvents();
  }

  bindEvents() {
    document.getElementById('start-btn')?.addEventListener('click', () => this.startGame());
    document.getElementById('yes-btn')?.addEventListener('click', () => this.answer(true));
    document.getElementById('no-btn')?.addEventListener('click', () => this.answer(false));
    document.getElementById('skip-btn')?.addEventListener('click', () => this.skip());
    document.getElementById('retry-btn')?.addEventListener('click', () => this.restart());
    document.getElementById('reload-btn')?.addEventListener('click', () => location.reload());
    document.getElementById('credits-link')?.addEventListener('click', (e) => {
      e.preventDefault();
      this.elements.creditsModal.classList.remove('hidden');
    });
    document.getElementById('close-credits')?.addEventListener('click', () => {
      this.elements.creditsModal.classList.add('hidden');
    });
  }

  async init() {
    try {
      await this.loadData();
      this.showScreen('start');
    } catch (error) {
      console.error('Failed to load data:', error);
      this.elements.errorMessage.textContent = `データの読み込みに失敗しました: ${error.message}`;
      this.showScreen('error');
    }
  }

  async loadData() {
    // Try to load from docs/ (GitHub Pages) or local data/
    const basePaths = ['../docs/', '../data/', './'];
    let loaded = false;

    for (const basePath of basePaths) {
      try {
        const indexRes = await fetch(`${basePath}index.json`);
        if (!indexRes.ok) continue;

        const index = await indexRes.json();

        // Load taxonomy
        const taxonomyRes = await fetch(`${basePath}${index.taxonomy || 'taxonomy.jsonl'}`);
        if (taxonomyRes.ok) {
          const taxonomyText = await taxonomyRes.text();
          this.parseTaxonomy(taxonomyText);
        }

        // Load questions
        const questionsRes = await fetch(`${basePath}${index.questions || 'questions.jsonl'}`);
        if (questionsRes.ok) {
          const questionsText = await questionsRes.text();
          this.parseQuestions(questionsText);
        }

        loaded = true;
        break;
      } catch (e) {
        continue;
      }
    }

    if (!loaded) {
      // Try loading JSONL files directly
      for (const basePath of basePaths) {
        try {
          const taxonomyRes = await fetch(`${basePath}taxonomy.jsonl`);
          const questionsRes = await fetch(`${basePath}questions.jsonl`);

          if (taxonomyRes.ok && questionsRes.ok) {
            this.parseTaxonomy(await taxonomyRes.text());
            this.parseQuestions(await questionsRes.text());
            loaded = true;
            break;
          }
        } catch (e) {
          continue;
        }
      }
    }

    if (!loaded || this.taxonomy.size === 0) {
      throw new Error('分類データが見つかりません');
    }

    // Build children map
    this.buildChildrenMap();
  }

  parseTaxonomy(text) {
    const lines = text.trim().split('\n').filter(l => l.trim());
    for (const line of lines) {
      try {
        const node = JSON.parse(line);
        this.taxonomy.set(node.taxonID, node);
      } catch (e) {
        console.warn('Failed to parse taxonomy line:', line);
      }
    }
  }

  parseQuestions(text) {
    const lines = text.trim().split('\n').filter(l => l.trim());
    for (const line of lines) {
      try {
        const q = JSON.parse(line);
        const nodeId = q.node_id;
        if (!this.questions.has(nodeId)) {
          this.questions.set(nodeId, []);
        }
        this.questions.get(nodeId).push(q);
      } catch (e) {
        console.warn('Failed to parse question line:', line);
      }
    }
  }

  buildChildrenMap() {
    // Initialize all nodes with empty children arrays
    for (const [id, node] of this.taxonomy) {
      if (!this.childrenMap.has(id)) {
        this.childrenMap.set(id, []);
      }
    }

    // Build parent-child relationships
    for (const [id, node] of this.taxonomy) {
      if (node.parentID && this.taxonomy.has(node.parentID)) {
        if (!this.childrenMap.has(node.parentID)) {
          this.childrenMap.set(node.parentID, []);
        }
        this.childrenMap.get(node.parentID).push(id);
      }
    }
  }

  showScreen(name) {
    Object.values(this.screens).forEach(s => s?.classList.add('hidden'));
    this.screens[name]?.classList.remove('hidden');
  }

  startGame() {
    // Find root node (Animalia or the topmost node)
    let rootId = 'Animalia';
    if (!this.taxonomy.has(rootId)) {
      // Find a node without parent
      for (const [id, node] of this.taxonomy) {
        if (!node.parentID || !this.taxonomy.has(node.parentID)) {
          rootId = id;
          break;
        }
      }
    }

    const root = this.taxonomy.get(rootId);
    const children = this.childrenMap.get(rootId) || [];

    this.state = {
      currentNode: rootId,
      candidates: this.getAllDescendants(rootId),
      path: [rootId],
      usedQuestions: new Set(),
    };

    this.questionCount = 0;
    this.history = [];

    this.updateBreadcrumb();
    this.nextQuestion();
    this.showScreen('game');
  }

  getAllDescendants(nodeId) {
    const descendants = new Set();
    const stack = [nodeId];

    while (stack.length > 0) {
      const current = stack.pop();
      const children = this.childrenMap.get(current) || [];

      if (children.length === 0) {
        // Leaf node
        descendants.add(current);
      } else {
        stack.push(...children);
      }
    }

    // If no leaves found, add the node itself
    if (descendants.size === 0) {
      descendants.add(nodeId);
    }

    return descendants;
  }

  getSubtreeTaxa(subtreeId) {
    // Handle special "minus" subtrees
    if (subtreeId.includes('_minus_')) {
      const parts = subtreeId.split('_minus_');
      const baseId = parts[0];
      const excludeIds = parts.slice(1);

      const base = this.getAllDescendants(baseId);
      for (const excludeId of excludeIds) {
        const exclude = this.getAllDescendants(excludeId);
        exclude.forEach(id => base.delete(id));
      }
      return base;
    }

    return this.getAllDescendants(subtreeId);
  }

  nextQuestion() {
    // Check win/lose conditions
    if (this.state.candidates.size === 1) {
      const winner = Array.from(this.state.candidates)[0];
      this.showResult(winner, true);
      return;
    }

    if (this.state.candidates.size === 0) {
      this.showResult(null, false);
      return;
    }

    // Find the next question
    const question = this.findBestQuestion();

    if (!question) {
      // No more questions, show best guess
      const bestGuess = this.getBestGuess();
      this.showResult(bestGuess, false);
      return;
    }

    this.currentQuestion = question;
    this.displayQuestion(question);
  }

  findBestQuestion() {
    // Look for questions at current node first, then traverse up/down
    const nodesToCheck = [this.state.currentNode];

    // Add ancestors
    let node = this.taxonomy.get(this.state.currentNode);
    while (node && node.parentID) {
      nodesToCheck.push(node.parentID);
      node = this.taxonomy.get(node.parentID);
    }

    // Add descendant nodes that are in candidates
    for (const candidateId of this.state.candidates) {
      let current = this.taxonomy.get(candidateId);
      while (current) {
        if (!nodesToCheck.includes(current.taxonID)) {
          nodesToCheck.push(current.taxonID);
        }
        if (!current.parentID) break;
        current = this.taxonomy.get(current.parentID);
      }
    }

    // Find best question (most balanced split)
    let bestQuestion = null;
    let bestScore = -1;

    for (const nodeId of nodesToCheck) {
      const nodeQuestions = this.questions.get(nodeId) || [];

      for (const q of nodeQuestions) {
        if (this.state.usedQuestions.has(q.q_id)) continue;

        // Calculate how well this question splits candidates
        const yesSet = this.getSubtreeTaxa(q.yes_next);
        const noSet = this.getSubtreeTaxa(q.no_next);

        let yesCount = 0, noCount = 0;
        for (const c of this.state.candidates) {
          if (yesSet.has(c)) yesCount++;
          if (noSet.has(c)) noCount++;
        }

        // Skip if question doesn't split candidates
        if (yesCount === 0 || noCount === 0) continue;
        if (yesCount + noCount < this.state.candidates.size * 0.5) continue;

        // Score: prefer balanced splits
        const total = yesCount + noCount;
        const balance = Math.min(yesCount, noCount) / Math.max(yesCount, noCount);
        const coverage = total / this.state.candidates.size;
        const score = balance * coverage * (q.confidence || 0.5);

        if (score > bestScore) {
          bestScore = score;
          bestQuestion = q;
        }
      }
    }

    return bestQuestion;
  }

  displayQuestion(question) {
    this.questionCount++;
    this.elements.questionNum.textContent = this.questionCount;
    this.elements.questionText.textContent = question.question;

    // Show confidence
    const confidence = question.confidence || 0.5;
    this.elements.confidence.textContent = `確信度: ${Math.round(confidence * 100)}%`;
    this.elements.confidence.classList.toggle('low', confidence < 0.8);

    // Show sources
    const sources = question.sources || [];
    this.elements.sources.textContent = `出典: ${sources.join(', ') || '不明'}`;

    // Update candidates count
    this.elements.candidatesCount.textContent = this.state.candidates.size;
  }

  answer(isYes) {
    if (!this.currentQuestion) return;

    const q = this.currentQuestion;
    this.state.usedQuestions.add(q.q_id);

    // Record history
    this.history.push({
      question: q,
      answer: isYes,
      candidatesBefore: new Set(this.state.candidates),
    });

    // Filter candidates
    const targetSubtree = isYes ? q.yes_next : q.no_next;
    const targetTaxa = this.getSubtreeTaxa(targetSubtree);

    const newCandidates = new Set();
    for (const c of this.state.candidates) {
      if (targetTaxa.has(c)) {
        newCandidates.add(c);
      }
    }

    // If filtering removes all candidates, keep at least some
    if (newCandidates.size === 0) {
      // Question didn't apply well, keep original candidates
      console.warn('Question filtered all candidates, keeping original');
    } else {
      this.state.candidates = newCandidates;
    }

    // Update current node if we've narrowed to a subtree
    this.updateCurrentNode();
    this.updateBreadcrumb();
    this.nextQuestion();
  }

  skip() {
    if (!this.currentQuestion) return;

    // Mark as used but don't filter
    this.state.usedQuestions.add(this.currentQuestion.q_id);
    this.nextQuestion();
  }

  updateCurrentNode() {
    // Find the lowest common ancestor of remaining candidates
    if (this.state.candidates.size === 0) return;

    const candidates = Array.from(this.state.candidates);

    // Get path to root for first candidate
    const paths = candidates.map(c => this.getPathToRoot(c));

    // Find lowest common ancestor
    let lca = null;
    const firstPath = paths[0];

    for (let i = firstPath.length - 1; i >= 0; i--) {
      const nodeId = firstPath[i];
      const isCommon = paths.every(p => p.includes(nodeId));
      if (isCommon) {
        lca = nodeId;
        break;
      }
    }

    if (lca && lca !== this.state.currentNode) {
      this.state.currentNode = lca;
      this.state.path = this.getPathToRoot(lca).reverse();
    }
  }

  getPathToRoot(nodeId) {
    const path = [];
    let current = nodeId;

    while (current) {
      path.push(current);
      const node = this.taxonomy.get(current);
      if (!node || !node.parentID) break;
      current = node.parentID;
    }

    return path;
  }

  updateBreadcrumb() {
    const path = this.state.path;
    const crumbs = path.map(nodeId => {
      const node = this.taxonomy.get(nodeId);
      const name = node?.vernacularName || node?.scientificName || nodeId;
      return `<span class="crumb">${this.escapeHtml(name)}</span>`;
    });

    this.elements.breadcrumb.innerHTML = crumbs.join('');
  }

  getBestGuess() {
    // Return the most likely candidate (first one, or random)
    const candidates = Array.from(this.state.candidates);
    return candidates[0] || null;
  }

  showResult(taxonId, isSuccess) {
    const card = this.elements.resultCard;
    const title = this.elements.resultTitle;
    const content = this.elements.resultContent;

    card.className = 'result-card ' + (isSuccess ? 'success' : 'unknown');

    if (isSuccess && taxonId) {
      const node = this.taxonomy.get(taxonId);
      title.textContent = '正解!';

      const path = this.getPathToRoot(taxonId).reverse();
      const pathStr = path.map(id => {
        const n = this.taxonomy.get(id);
        return n?.scientificName || id;
      }).join(' > ');

      content.innerHTML = `
        <div class="result-content">
          <div class="taxon-name">${this.escapeHtml(node?.vernacularName || node?.scientificName || taxonId)}</div>
          <div class="scientific-name">${this.escapeHtml(node?.scientificName || '')}</div>
          <div class="taxonomy-path">${this.escapeHtml(pathStr)}</div>
          <div class="sources-info">
            ${this.questionCount}問で特定しました
          </div>
        </div>
      `;
    } else if (taxonId) {
      const node = this.taxonomy.get(taxonId);
      title.textContent = '推測';

      content.innerHTML = `
        <div class="result-content">
          <div class="taxon-name">${this.escapeHtml(node?.vernacularName || node?.scientificName || taxonId)}?</div>
          <div class="scientific-name">${this.escapeHtml(node?.scientificName || '')}</div>
          <div class="sources-info">
            設問が不足しています（残り候補: ${this.state.candidates.size}）<br>
            データの拡充にご協力ください
          </div>
        </div>
      `;
    } else {
      title.textContent = '特定できませんでした';
      content.innerHTML = `
        <div class="result-content">
          <div class="sources-info">
            該当する動物が見つかりませんでした。<br>
            データに含まれていない可能性があります。
          </div>
        </div>
      `;
    }

    this.showScreen('result');
  }

  restart() {
    this.startGame();
  }

  escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}

// Initialize game when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const game = new Animal20Q();
  game.init();
});
