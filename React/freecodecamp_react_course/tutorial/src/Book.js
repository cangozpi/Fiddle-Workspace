import React from 'react'


const Book = ({ img, title, author }) => {
  // const {img, title, author} = props
  const clickHandler = (e) => {
    console.log(e);
    console.log(e.target);
    alert("hello world");
  };
  const complexExample = (author) => {
    console.log(author);
  };
  return (
    <article
      className="book"
      onMouseOver={() => {
        console.log(title);
      }}
    >
      <img src={img} alt="Book cover" />
      <h1
        onClick={() => {
          console.log(title);
        }}
      >
        {title}
      </h1>
      <h4 style={{ color: "#617d98", fontSize: "0.75rem", margin: "0.25rem" }}>
        {author}
      </h4>
      <button type="button" onClick={clickHandler}>
        {" "}
        reference example
      </button>
      <button type="button" onClick={() => complexExample(author)}>
        {" "}
        more complex example
      </button>
    </article>
  );
};


export default Book