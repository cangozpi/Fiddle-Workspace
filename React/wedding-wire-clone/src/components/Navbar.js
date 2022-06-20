import React from "react";
import "../styles/Navbar.css";
import logo from "../logo.svg";

function Navbar() {
  return (
    <div className="nav">
      <div className="logo">
        <img src={logo} alt="Wedding Wire Logo" />
      </div>
      <div className="middle_nav">
        <div>
          <h2>plannign tools</h2>
        </div>
        <div>
          <h2>Venues</h2>
        </div>
        <div>
          <h2>vendors</h2>
        </div>
        <div>
          <h2>forums</h2>
        </div>
        <div>
          <h2>dresses</h2>
        </div>
        <div>
          <h2>ideas</h2>
        </div>
        <div>
          <h2>registry</h2>
        </div>
        <div>
          <h2>wedding website</h2>
        </div>
        <div>
          <h2>invitations</h2>
        </div>
        <div>
          <h2>shop</h2>
        </div>
      </div>
      <div className="right_nav">
        <div className="question">
          <i class="bi bi-briefcase"></i>
          <h2>are you a vendor?</h2>
        </div>
        <div className="authentication_section">
          <h2>log in</h2>
          <h2>join now</h2>
        </div>
      </div>
    </div>
  );
}

export default Navbar;
