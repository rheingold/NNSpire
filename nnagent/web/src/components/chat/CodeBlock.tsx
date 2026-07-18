/**
 * CodeBlock — Renders code blocks with syntax highlighting and copy-to-clipboard.
 *
 * Uses highlight.js for syntax highlighting. Supports copy button
 * that provides feedback on success/failure.
 */

import React, { useEffect, useRef, useState } from 'react'
import hljs from 'highlight.js/lib/core'
import javascript from 'highlight.js/lib/languages/javascript'
import typescript from 'highlight.js/lib/languages/typescript'
import python from 'highlight.js/lib/languages/python'
import jsonLang from 'highlight.js/lib/languages/json'
import xml from 'highlight.js/lib/languages/xml'
import css from 'highlight.js/lib/languages/css'
import markdown from 'highlight.js/lib/languages/markdown'
import bash from 'highlight.js/lib/languages/bash'
import sql from 'highlight.js/lib/languages/sql'
import cpp from 'highlight.js/lib/languages/cpp'
import rust from 'highlight.js/lib/languages/rust'
import go from 'highlight.js/lib/languages/go'
import java from 'highlight.js/lib/languages/java'
import 'highlight.js/styles/atom-one-dark.css'

// Register common languages
hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('js', javascript)
hljs.registerLanguage('typescript', typescript)
hljs.registerLanguage('ts', typescript)
hljs.registerLanguage('python', python)
hljs.registerLanguage('py', python)
hljs.registerLanguage('json', jsonLang)
hljs.registerLanguage('xml', xml)
hljs.registerLanguage('html', xml)
hljs.registerLanguage('css', css)
hljs.registerLanguage('markdown', markdown)
hljs.registerLanguage('md', markdown)
hljs.registerLanguage('bash', bash)
hljs.registerLanguage('sh', bash)
hljs.registerLanguage('shell', bash)
hljs.registerLanguage('sql', sql)
hljs.registerLanguage('cpp', cpp)
hljs.registerLanguage('c++', cpp)
hljs.registerLanguage('rust', rust)
hljs.registerLanguage('rs', rust)
hljs.registerLanguage('go', go)
hljs.registerLanguage('golang', go)
hljs.registerLanguage('java', java)

interface CodeBlockProps {
  language: string
  content: string
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ language, content }) => {
  const codeRef = useRef<HTMLElement>(null)
  const [copied, setCopied] = useState(false)

  // Highlight code on mount
  useEffect(() => {
    if (codeRef.current) {
      try {
        // Try to highlight with specified language, fallback to auto-detect
        const lang = language.toLowerCase()
        if (hljs.getLanguage(lang)) {
          hljs.highlightElement(codeRef.current)
        } else {
          hljs.highlightElement(codeRef.current)
        }
      } catch (error) {
        console.warn(`[CodeBlock] Syntax highlighting failed for language "${language}":`, error)
      }
    }
  }, [language, content])

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      console.error('[CodeBlock] Copy to clipboard failed:', error)
      // Fallback for older browsers
      try {
        const textArea = document.createElement('textarea')
        textArea.value = content
        document.body.appendChild(textArea)
        textArea.select()
        document.execCommand('copy')
        document.body.removeChild(textArea)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      } catch (fallbackError) {
        console.error('[CodeBlock] Fallback copy also failed:', fallbackError)
      }
    }
  }

  const displayLanguage = language || 'text'

  return (
    <div className="code-block">
      <div className="code-block-header">
        <span className="code-block-language">{displayLanguage}</span>
        <button
          className="code-block-copy"
          onClick={handleCopy}
          aria-label="Copy code to clipboard"
          title="Copy to clipboard"
        >
          {copied ? '✓ Copied' : '📋 Copy'}
        </button>
      </div>
      <pre className="code-block-pre">
        <code
          ref={codeRef}
          className={`language-${language.toLowerCase()}`}
        >
          {content}
        </code>
      </pre>
    </div>
  )
}

export default CodeBlock
