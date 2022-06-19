import React from "react";
import "./styles/Post.css";
import { Avatar } from "@mui/material";
import VerifiedOutlinedIcon from "@mui/icons-material/VerifiedOutlined";
import ChatBubbleOutlineOutlinedIcon from "@mui/icons-material/ChatBubbleOutlineOutlined";
import RepeatOutlinedIcon from "@mui/icons-material/RepeatOutlined";
import FavoriteBorderOutlinedIcon from "@mui/icons-material/FavoriteBorderOutlined";
import PublishOutlinedIcon from "@mui/icons-material/PublishOutlined";

function Post({ displayName, username, verified, text, image, avatar }) {
  return (
    <div className="post">
      <div className="post__avatar">
        <Avatar src="https://cdn4.iconfinder.com/data/icons/superheroes/512/ironman-512.png"></Avatar>
      </div>
      <div className="post__body">
        <div className="post__header">
          <div className="post__headerText">
            <h3>
              Tony Stark{" "}
              <span className="post__headerSpecial">
                <VerifiedOutlinedIcon className="post__badge" />
              </span>
            </h3>
          </div>
          <div className="post__headerDescription">
            <p>Do not throw another moon at me !</p>
          </div>
        </div>
        <img
          src="https://data.whicdn.com/images/329852833/original.gif"
          alt=""
        />
        <div className="post__footer">
          <ChatBubbleOutlineOutlinedIcon fontSize="small" />
          <RepeatOutlinedIcon fontSize="small" />
          <FavoriteBorderOutlinedIcon fontSize="small" />
          <PublishOutlinedIcon fontSize="small" />
        </div>
      </div>
    </div>
  );
}

export default Post;
