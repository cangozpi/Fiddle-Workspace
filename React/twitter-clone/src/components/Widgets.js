import React from "react";
import "./styles/Widgets.css";
import {
  TwitterTimelineEmbed,
  TwitterShareButton,
  TwitterTweetEmbed,
} from "react-twitter-embed";
import SearchOutlinedIcon from "@mui/icons-material/SearchOutlined";

function Widgets() {
  return (
    <div className="widgets">
      <div className="widgets__input">
        <SearchOutlinedIcon />
        <input type="text" placeholder="Search Twitter" />
      </div>

      <div className="widgets__widgetContainer">
        <h2>What's happening</h2>
        <TwitterTweetEmbed tweetId={"1538208757905297409"} />
        <TwitterTimelineEmbed
          sourceType="profile"
          screenName="elonmusk"
          options={{ height: 400 }}
        />

        <TwitterShareButton
          url="https://twitter.com/robertdowneyjr"
          options={{
            text: "Iron man > Thanos",
            via: "robertdowneyjr",
          }}
        />
      </div>
    </div>
  );
}

export default Widgets;
