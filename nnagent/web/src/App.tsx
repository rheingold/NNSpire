import React, { useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { ChatProvider } from './context/ChatContext'
import { ChatWindow } from './components/chat/ChatWindow'
import { ConversationSidebar } from './components/chat/ConversationSidebar'
import './App.css'
import './components/chat/Chat.css'
import './components/chat/ConversationSidebar.css'

// ─── Chat Page (wraps ChatWindow with conversation management) ────────────────

const ChatPage: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <div className="chat-page">
      {/* Sidebar */}
      {sidebarOpen && (
        <aside className="chat-sidebar">
          <ConversationSidebar />
        </aside>
      )}

      {/* Toggle Sidebar Button */}
      <button
        className="sidebar-toggle"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
      >
        {sidebarOpen ? '◀' : '▶'}
      </button>

      {/* Main Chat Area */}
      <main className="chat-main">
        <ChatWindow />
      </main>
    </div>
  )
}

// ─── Placeholder Pages ────────────────────────────────────────────────────────

const Home: React.FC = () => (
  <div className="home-page">
    <h1>NNSpire Agent</h1>
    <p className="version">v0.2.0 — PH2 Chat Interface</p>
    <p className="welcome">
      Welcome to NNSpire Agent. Start a new conversation to begin chatting.
    </p>
  </div>
)

const Settings: React.FC = () => (
  <div className="settings-page">
    <h2>Settings</h2>
    <p>Configuration panel under construction.</p>
  </div>
)

const About: React.FC = () => (
  <div className="about-page">
    <h2>About</h2>
    <p>NNSpire Agent — Conversational AI Workbench</p>
    <p>Part of the NNSpire project.</p>
  </div>
)

// ─── App Shell ────────────────────────────────────────────────────────────────

const AppRoutes: React.FC = () => {
  return (
    <div className="app-shell">
      <header className="app-header">
        <nav className="app-nav">
          <a href="/">Home</a>
          <a href="/chat">Chat</a>
          <a href="/settings">Settings</a>
          <a href="/about">About</a>
        </nav>
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer className="app-footer">
        <span>NNSpire Agent © {new Date().getFullYear()}</span>
      </footer>
    </div>
  )
}

const App: React.FC = () => {
  return (
    <ChatProvider>
      <AppRoutes />
    </ChatProvider>
  )
}

export default App
