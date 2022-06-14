import React from 'react'
import ReactDom from 'react-dom'
// CSS
import './index.css'

import { books } from './books'
import Book from './Book'


// const BookList = () => {
//   return React.createElement('div', {}, React.createElement('h1', {}, 'Hello World'));
// }


function BookList(){
  return (
    <section className="booklist">
      {books.map((book) => {
        return <Book key={book.id} {...book}></Book>;
      })}
    </section>
  );  
}


// Inject BookList DOM object into the element with id:'root' in the index.html file in the public folder which is the entry
// point to our web application
ReactDom.render(<BookList />, document.getElementById("root"));