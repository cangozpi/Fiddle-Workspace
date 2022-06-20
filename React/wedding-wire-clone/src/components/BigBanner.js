import React from "react";
import "../styles/BigBanner.css";

function BigBanner() {
  return (
    <div className="big-banner">
      <div className="container-fluid g-0 h-100">
        <div className="row justify-content-center justify-content-lg-end text-left h-100 g-0">
          <div className=" text-center text-lg-center left order-2 order-lg-1 col-lg-5 col-10 col-md-10 d-flex flex-column justify-content-center align-items-left">
            <h2>Let's find your wedding team</h2>
            <p>
              Search over 250,000 local professionals with reviews, pricing,
              availability, and more
            </p>
            <div class="input-group">
              <div className="input-group-text" id="btnGroupAddon">
                <i class="bi bi-search"></i>
              </div>
              <input
                type="text"
                className="form-control"
                placeholder="Start you search"
                aria-label="Input group example"
                aria-describedby="btnGroupAddon"
              />
              <input
                type="text"
                className="form-control"
                placeholder="in Where"
                aria-label="Input group example"
                aria-describedby="btnGroupAddon"
              />
              <button type="button" class="btn btn-info">
                Search
              </button>
            </div>
          </div>
          <div className="order-1 order-lg-2 right col-11 col-lg-6"></div>
        </div>
      </div>
    </div>
  );
}

export default BigBanner;
