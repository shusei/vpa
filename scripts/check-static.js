#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, '..');

const DEFAULT_HTML_FILES = ['index.html'];
const DEFAULT_CSS_FILES = [
  'assets/css/base.css',
  'assets/css/layout.css',
  'assets/css/components.css',
  'assets/css/overlays.css',
];
const THREAD_CONFIG_FILES = ['assets/app-core.js', 'assets/js/app-bootstrap.js'];

function readSource(relativePath) {
  const filePath = resolve(projectRoot, relativePath);
  try {
    return readFileSync(filePath, 'utf8');
  } catch (error) {
    return { error: `無法讀取 ${relativePath}：${error.message}` };
  }
}

function checkHTML(relativePath) {
  const sourceOrError = readSource(relativePath);
  if (sourceOrError?.error) {
    return [sourceOrError.error];
  }

  const source = sourceOrError;
  const cleaned = source
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/<!doctype[^>]*>/gi, '');

  const issues = [];
  const tagRegex = /<\s*(\/)?\s*([a-zA-Z0-9:-]+)([^>]*)>/g;
  const voidTags = new Set([
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
    'keygen', 'link', 'meta', 'param', 'source', 'track', 'wbr'
  ]);
  const stack = [];

  let match;
  while ((match = tagRegex.exec(cleaned))) {
    const isClosing = Boolean(match[1]);
    const tagName = match[2].toLowerCase();
    const full = match[0];

    if (tagName.startsWith('!')) {
      continue;
    }

    const selfClosing = /\/\s*>$/.test(full) || voidTags.has(tagName);

    if (!isClosing) {
      if (!selfClosing) {
        stack.push({ tag: tagName, index: match.index });
      }
      continue;
    }

    if (!stack.length) {
      issues.push(`無法為 </${tagName}> 找到對應的開啟標籤（字元索引 ${match.index}）。`);
      continue;
    }

    const last = stack.pop();
    if (last.tag !== tagName) {
      issues.push(`標籤 <${last.tag}> 與 </${tagName}> 不成對（關閉標籤索引 ${match.index}）。`);
    }
  }

  for (const unmatched of stack) {
    issues.push(`標籤 <${unmatched.tag}> 未被關閉（開啟標籤索引 ${unmatched.index}）。`);
  }

  return issues;
}

function checkCSS(relativePath) {
  const sourceOrError = readSource(relativePath);
  if (sourceOrError?.error) {
    return [sourceOrError.error];
  }

  const source = sourceOrError;
  const issues = [];

  let braces = 0;
  let brackets = 0;
  let parens = 0;
  let line = 1;
  let column = 0;
  let inString = null;
  let escape = false;
  let inComment = false;

  const updatePosition = (ch) => {
    if (ch === '\n') {
      line += 1;
      column = 0;
    } else {
      column += 1;
    }
  };

  for (let i = 0; i < source.length; i += 1) {
    const ch = source[i];
    const next = source[i + 1];

    if (inComment) {
      if (ch === '*' && next === '/') {
        inComment = false;
        i += 1;
        updatePosition('*');
        updatePosition('/');
        continue;
      }
      updatePosition(ch);
      continue;
    }

    if (inString) {
      if (escape) {
        escape = false;
      } else if (ch === '\\') {
        escape = true;
      } else if (ch === inString) {
        inString = null;
      }
      updatePosition(ch);
      continue;
    }

    if (ch === '\'' || ch === '"') {
      inString = ch;
      updatePosition(ch);
      continue;
    }

    if (ch === '/' && next === '*') {
      inComment = true;
      updatePosition('/');
      updatePosition('*');
      i += 1;
      continue;
    }

    switch (ch) {
      case '{':
        braces += 1;
        break;
      case '}':
        braces -= 1;
        if (braces < 0) {
          issues.push(`在第 ${line} 行第 ${column + 1} 列遇到多餘的 '}'。`);
          braces = 0;
        }
        break;
      case '[':
        brackets += 1;
        break;
      case ']':
        brackets -= 1;
        if (brackets < 0) {
          issues.push(`在第 ${line} 行第 ${column + 1} 列遇到多餘的 ']'。`);
          brackets = 0;
        }
        break;
      case '(':
        parens += 1;
        break;
      case ')':
        parens -= 1;
        if (parens < 0) {
          issues.push(`在第 ${line} 行第 ${column + 1} 列遇到多餘的 ')'。`);
          parens = 0;
        }
        break;
      default:
        break;
    }

    updatePosition(ch);
  }

  if (inString) {
    issues.push('CSS 結束時仍處於字串內，請檢查引號是否閉合。');
  }

  if (inComment) {
    issues.push('CSS 結束時仍處於多行註解內，請檢查 /* ... */ 是否配對。');
  }

  if (braces > 0) {
    issues.push(`CSS 中仍有 ${braces} 個未關閉的 '{'。`);
  }
  if (brackets > 0) {
    issues.push(`CSS 中仍有 ${brackets} 個未關閉的 '['。`);
  }
  if (parens > 0) {
    issues.push(`CSS 中仍有 ${parens} 個未關閉的 '('。`);
  }

  return issues;
}

function checkThreadConfig(relativePath) {
  const sourceOrError = readSource(relativePath);
  if (sourceOrError?.error) {
    return [sourceOrError.error];
  }

  const source = sourceOrError;
  const assignmentRegex = /env\.backends\.onnx\.wasm\.numThreads\s*=\s*([A-Za-z_$][\w$]*)/;
  const match = assignmentRegex.exec(source);

  if (!match) {
    return ['未找到 env.backends.onnx.wasm.numThreads 的設定，請確認初始化時已指定執行緒數。'];
  }

  const identifier = match[1];
  const declarationRegex = new RegExp(`\\b(?:const|let|var)\\s+${identifier}\\s*=`);

  if (!declarationRegex.test(source)) {
    return [`找不到 ${identifier} 的宣告，請確認指定給 numThreads 的變數已定義。`];
  }

  return [];
}

const args = process.argv.slice(2);
const extras = {
  html: new Set(),
  css: new Set(),
  unsupported: [],
};

for (const arg of args) {
  const lower = arg.toLowerCase();
  if (lower.endsWith('.html') || lower.endsWith('.htm')) {
    extras.html.add(arg);
  } else if (lower.endsWith('.css')) {
    extras.css.add(arg);
  } else {
    extras.unsupported.push(arg);
  }
}

if (extras.unsupported.length) {
  console.error(
    `無法判斷下列檔案的型別：${extras.unsupported.join(', ')}。` +
      '請提供 .html / .htm 或 .css 檔案名稱。'
  );
  process.exit(1);
}

const htmlTargets = new Set(DEFAULT_HTML_FILES);
const cssTargets = new Set(DEFAULT_CSS_FILES);

extras.html.forEach((file) => htmlTargets.add(file));
extras.css.forEach((file) => cssTargets.add(file));

const allIssues = [];

for (const file of htmlTargets) {
  const issues = checkHTML(file);
  if (issues.length) {
    allIssues.push(`HTML 檢查失敗（${file}）：`);
    allIssues.push(...issues.map((issue) => `  - ${issue}`));
  } else {
    console.log(`[OK] ${file}：HTML 結構檢查通過。`);
  }
}

for (const file of cssTargets) {
  const issues = checkCSS(file);
  if (issues.length) {
    allIssues.push(`CSS 檢查失敗（${file}）：`);
    allIssues.push(...issues.map((issue) => `  - ${issue}`));
  } else {
    console.log(`[OK] ${file}：CSS 結構檢查通過。`);
  }
}

let threadConfigPassed = false;
const threadConfigErrors = [];
for (const file of THREAD_CONFIG_FILES) {
  const issues = checkThreadConfig(file);
  if (!issues.length) {
    console.log(`[OK] ${file}：WASM 執行緒設定檢查通過。`);
    threadConfigPassed = true;
    break;
  }
  threadConfigErrors.push({ file, issues });
}

if (!threadConfigPassed) {
  allIssues.push(`WASM 執行緒設定檢查失敗（${THREAD_CONFIG_FILES.join(' / ')}）：`);
  threadConfigErrors.forEach(({ file, issues }) => {
    allIssues.push(`  - ${file}`);
    allIssues.push(...issues.map((issue) => `    - ${issue}`));
  });
}

if (allIssues.length) {
  console.error(allIssues.join('\n'));
  process.exitCode = 1;
} else {
  console.log('靜態資源語法檢查完成。');
}
