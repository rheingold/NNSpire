import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import App from './App'

test('renders app shell with chat header and user menu', () => {
  render(
    <BrowserRouter>
      <App />
    </BrowserRouter>
  )

  // Chat is the default home/dashboard page
  expect(screen.getByText(/NNSpire Chat/i)).toBeInTheDocument()
  // User menu button for navigation (bottom-left)
  expect(screen.getByRole('button', { name: /User menu/i })).toBeInTheDocument()
  // Footer copyright
  expect(screen.getByText(/NNSpire Agent \©/i)).toBeInTheDocument()
})
