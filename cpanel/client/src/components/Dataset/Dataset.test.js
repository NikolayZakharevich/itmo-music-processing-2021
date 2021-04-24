import { render, screen } from '@testing-library/react';
import Dataset from './Dataset';

test('renders learn react link', () => {
  render(<Dataset />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});
