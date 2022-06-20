class Data {
  constructor(caption, src) {
    this.data = {
      caption: caption,
      src: src,
    };
  }
}

const content = [
  new Data(
    "Wedding Venues",
    "https://cdn1.weddingwire.com/assets/svg/original/illustration/search.svg"
  ),
  new Data(
    "Invitations",
    "https://cdn1.weddingwire.com/assets/svg/original/illustration/envelope.svg"
  ),
  new Data(
    "Wedding Website",
    "https://cdn1.weddingwire.com/assets/svg/original/illustration/websites.svg"
  ),
  new Data(
    "Planner",
    "https://cdn1.weddingwire.com/assets/svg/original/illustration/notebook.svg"
  ),
  new Data(
    "Ideas",
    "https://cdn1.weddingwire.com/assets/svg/original/illustration/lightbulb.svg"
  ),
  new Data(
    "Wedding Dresses",
    "https://cdn1.weddingwire.com/assets/svg/original/illustration/dress.svg"
  ),
];

export default content;
