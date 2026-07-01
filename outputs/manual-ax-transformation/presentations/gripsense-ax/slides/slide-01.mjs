import { makeSlide } from "./deck.mjs";
export async function slide01(presentation, ctx) {
  return makeSlide(presentation, ctx, 1);
}
