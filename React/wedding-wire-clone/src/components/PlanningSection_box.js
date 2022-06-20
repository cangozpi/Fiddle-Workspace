import React from "react";
import "../styles/PlanningSection_box.css";

function PlanningSection_box({ caption, src }) {
  return (
    <div className="col illustration_box mx-2 mt-4">
      <div className="illustration_box_wrapper">
        <div className="container">
          <div className="row">
            <div className="col d-flex justify-content-center">
              <img src={src} alt={caption + " illustration"} />
            </div>
          </div>
          <div className="row">
            <div className="col text-center">
              <h2 className="h2_hover">{caption}</h2>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PlanningSection_box;
