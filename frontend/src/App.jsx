import { useState } from "react";
import AdminPage from "./pages/AdminPage";
import HomePage from "./pages/HomePage";

export default function App() {
  const [page, setPage] = useState("home");

  if (page === "admin") {
    return <AdminPage onNavigateHome={() => setPage("home")} />;
  }
  return <HomePage onNavigateAdmin={() => setPage("admin")} />;
}
