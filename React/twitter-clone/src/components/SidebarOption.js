import React from "react";
import "./styles/SidebarOption.css";

function SidebarOption({ active, text, Icon }) {
  return (
    <div className={`sidebarOption ${active && "sidebarOption--active"}`}>
      {Icon && <Icon />}
      {text && <h2>{text}</h2>}
    </div>
  );
}

export default SidebarOption;
