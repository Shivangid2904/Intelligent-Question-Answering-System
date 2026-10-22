import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import AppShell from './components/layout/AppShell'
import OverviewPage from './pages/OverviewPage'
import LibraryPage from './pages/LibraryPage'
import ChatPage from './pages/ChatPage'
import MatrixPage from './pages/MatrixPage'
import HistoryPage from './pages/HistoryPage'
import StatusPage from './pages/StatusPage'
import { ROUTES } from './routes/routes'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path={ROUTES.overview} element={<OverviewPage />} />
          <Route path={ROUTES.library} element={<LibraryPage />} />
          <Route path={ROUTES.chat} element={<ChatPage />} />
          <Route path={ROUTES.matrix} element={<MatrixPage />} />
          <Route path={ROUTES.history} element={<HistoryPage />} />
          <Route path={ROUTES.status} element={<StatusPage />} />
          <Route path="*" element={<Navigate to={ROUTES.overview} replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
