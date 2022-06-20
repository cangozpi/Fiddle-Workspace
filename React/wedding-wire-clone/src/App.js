import React from "react";
import "./App.css";
import Navbar from "./components/Navbar";
import BigBanner from "./components/BigBanner";
import PlanningSection from "./components/PlanningSection";

function App() {
  return (
    <>
      <div className="App">
        <Navbar />
        <BigBanner />
        <PlanningSection />
        {/* Content */}
      </div>
    </>
  );
}

export default App;
