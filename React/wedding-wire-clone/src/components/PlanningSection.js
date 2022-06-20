import React from "react";
import "../styles/PlanningSection.css";
import content from "../contents/PlanningSection_contents";
import PlanningSection_box from "./PlanningSection_box";

function PlanningSection() {
  return (
    <div className="container-fluid px-5">
      <div className="row">
        <div className="col-3 illustration_box mx-2 mt-4">
          <div className="illustration_box_wrapper">
            <div className="container d-flex flex-column align-items-start">
              <div className="row">
                <div className="col text-center">
                  <h5>Easily plan your wedding</h5>
                </div>
              </div>
              <div className="row">
                <div className="col d-flex justify-content-center">
                  <h2 className="h2_link">Get started {">"}</h2>
                </div>
              </div>
            </div>
          </div>
        </div>
        {content.map((x) => {
          return (
            <PlanningSection_box caption={x.data.caption} src={x.data.src} />
          );
        })}
      </div>

      <div className="row">
        <div className="col justify-content-center d-flex mt-5">
          <p className="p_link">Find a couple's website or registry</p>
        </div>
      </div>
    </div>
  );
}

export default PlanningSection;
