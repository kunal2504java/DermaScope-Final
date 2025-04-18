import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import SignUp from './pages/SignUp';
import Login from './pages/Login';
import Home from './pages/home/Home';
import TakeATest from './pages/home/TakeATest';
import TestResults from './pages/home/TestResults';
import RootLayout from './pages/home/RootLayout';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-dark-bg text-text-primary">
        <Routes>
          <Route path="/signup" element={<SignUp />} />
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<RootLayout />}>
            <Route index element={<Home />} />
            <Route path="take-test" element={<TakeATest />} />
            <Route path="test-results" element={<TestResults />} />
          </Route>
        </Routes>
        <Toaster
          position="top-right"
          toastOptions={{
            className: 'bg-dark-card text-text-primary',
            style: {
              background: 'var(--dark-card)',
              color: 'var(--text-primary)',
            },
            success: {
              iconTheme: {
                primary: 'var(--success)',
                secondary: 'var(--dark-card)',
              },
            },
            error: {
              iconTheme: {
                primary: 'var(--error)',
                secondary: 'var(--dark-card)',
              },
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;
