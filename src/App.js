import React, { useState, useEffect } from 'react';
import { HashRouter as Router, Routes, Route, Navigate, Outlet, useNavigate } from 'react-router-dom';
import { SettingsProvider } from './SettingsContext';
import Login from './Login';
import Dashboard from './Dashboard';
import Medicare from './Medicare';
import Lifetrack from './Lifetrack';
import BInsight from './BInsight';
import RoomMatrix from './RoomMatrix';
import Profile from './Profile';
import Settings from './Settings';
import './App.css';
import backgroundImage from './alzheimer-care.png';

function MainAppLayout({ onLogout }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const navigate = useNavigate();

  return (
    <div className="app">
      <button className="menu-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>☰</button>
      <div className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <button onClick={() => navigate('/profile')}>👤 Profile</button>
        <button onClick={() => setMenuOpen(!menuOpen)}>📋 Menu {menuOpen ? '▲' : '▼'}</button>
        {menuOpen && (
          <div className="menu-options">
            <button onClick={() => navigate('/medicare')}>Medicare</button>
            <button onClick={() => navigate('/lifetrack')}>LifeTrack</button>
            <button onClick={() => navigate('/binsight')}>B-Insight</button>
            <button onClick={() => navigate('/roommatrix')}>Room Matrix</button>
          </div>
        )}
        <button onClick={() => navigate('/settings')}>⚙️ Settings</button>
        <button onClick={onLogout}>🚪 Logout</button>
      </div>

      <div style={{ padding: '1rem' }}>
        <Outlet />
      </div>
    </div>
  );
}

function App() {
  const [loggedIn, setLoggedIn] = useState(false);

  // Fetch from Glitch backend after login
  useEffect(() => {
    if (loggedIn) {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'https://plaid-calico-silk.glitch.me';

      fetch(`${backendUrl}/api/sensor-data`)
        .then((res) => res.json())
        .then((data) => {
          console.log('Fetched sensor data from backend:', data);
        })
        .catch((err) => {
          console.error('Failed to fetch sensor data:', err);
        });
    }
  }, [loggedIn]);

  return (
    <SettingsProvider>
      <Router>
        <Routes>
          {!loggedIn ? (
            <Route path="*" element={<Login onLogin={() => setLoggedIn(true)} />} />
          ) : (
            <>
              <Route path="/" element={<Navigate to="/menu" />} />
              <Route element={<MainAppLayout onLogout={() => setLoggedIn(false)} />}>
                <Route path="/menu" element={<Dashboard />} />
                <Route path="/medicare" element={<Medicare />} />
                <Route path="/lifetrack" element={<Lifetrack />} />
                <Route path="/binsight" element={<BInsight />} />
                <Route path="/roommatrix" element={<RoomMatrix />} />
                <Route path="/profile" element={<Profile />} />
                <Route path="/settings" element={<Settings />} />
              </Route>
            </>
          )}
        </Routes>
      </Router>
    </SettingsProvider>
  );
}

export default App;
