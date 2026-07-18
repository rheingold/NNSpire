import { render, screen } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import App from './App'

test('renders app shell with header and footer', () => {
  render(
    <BrowserRouter>
      <App />
    </BrowserRouter>
  )

  // Use role-based query to avoid ambiguity with multiple "NNSpire Agent" occurrences
  expect(screen.getByRole('heading', { name: /NNSpire Agent/i })).toBeInTheDocument()
  expect(screen.getByText(/Home/i)).toBeInTheDocument()
  expect(screen.getByText(/Settings/i)).toBeInTheDocument()
})
