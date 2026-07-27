import React, { useState, useCallback } from 'react'
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
  const [chatMenuOpen, setChatMenuOpen] = useState(false)
  const [sidebarWidth, setSidebarWidth] = useState(280)

  const toggleChatMenu = useCallback(() => setChatMenuOpen((prev) => !prev), [])

  // ─── Sidebar Resize Handler ─────────────────────────────────────────────
  const handleResizeStart = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault()
    const startX = e.clientX
    const startWidth = sidebarWidth

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const newWidth = startWidth + (moveEvent.clientX - startX)
      if (newWidth >= 180 && newWidth <= 600) {
        setSidebarWidth(newWidth)
      }
    }

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }, [sidebarWidth])

  return (
    <div className="chat-page">
      {/* Top-right 3-dot menu for current chat */}
      <div className="chat-header">
        <div className="chat-title-bar">
          <span className="chat-title">NNSpire Chat</span>
        </div>
        <div className="chat-menu-container">
          <button
            className="chat-menu-btn"
            onClick={toggleChatMenu}
            aria-label="Chat options"
            title="Chat options"
          >
            ⋮
          </button>
          {chatMenuOpen && (
            <div className="chat-menu-dropdown">
              <button className="chat-menu-item">Export Conversation</button>
              <button className="chat-menu-item">Clear Messages</button>
              <button className="chat-menu-item">Conversation Settings</button>
            </div>
          )}
        </div>
      </div>

      {/* Content area */}
      <div className="chat-content">
        {/* Sidebar */}
        {sidebarOpen && (
          <aside className="chat-sidebar" style={{ width: `${sidebarWidth}px` }}>
            <ConversationSidebar />
            {/* Resize Handle */}
            <div
              className="sidebar-resize-handle"
              onMouseDown={handleResizeStart}
            />
          </aside>
        )}

        {/* Toggle Sidebar Button */}
        <button
          className="sidebar-toggle"
          style={{ left: sidebarOpen ? `${sidebarWidth}px` : '0px' }}
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
    </div>
  )
}

// ─── Placeholder Pages ────────────────────────────────────────────────────────

const Profile: React.FC = () => (
  <div className="profile-page">
    <h2>Profile</h2>
    <p>User profile management. Coming soon.</p>
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
  const [userMenuOpen, setUserMenuOpen] = useState(false)

  const toggleUserMenu = useCallback(() => setUserMenuOpen((prev) => !prev), [])

  return (
    <div className="app-shell">
      {/* Main content — Chat is the default home/dashboard */}
      <main className="app-main">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      {/* User submenu — bottom-left corner */}
      <div className="app-user-area">
        <button
          className="user-menu-btn"
          onClick={toggleUserMenu}
          aria-label="User menu"
          title="Navigation"
        >
          ☰
        </button>
        {userMenuOpen && (
          <div className="user-menu-dropdown">
            <a href="/" className="user-menu-item" onClick={() => setUserMenuOpen(false)}>
              💬 Chat
            </a>
            <a href="/profile" className="user-menu-item" onClick={() => setUserMenuOpen(false)}>
              👤 Profile
            </a>
            <a href="/settings" className="user-menu-item" onClick={() => setUserMenuOpen(false)}>
              ⚙️ Settings
            </a>
            <a href="/about" className="user-menu-item" onClick={() => setUserMenuOpen(false)}>
              ℹ️ About
            </a>
          </div>
        )}
      </div>

      {/* Footer */}
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
