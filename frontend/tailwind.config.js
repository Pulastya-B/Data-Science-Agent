/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./*.tsx",
    "./components/*.tsx",
    "./lib/*.ts",
  ],
  theme: {
    extend: {
      animation: {
        'bounce': 'bounce 1s infinite',
      },
    },
  },
  plugins: [],
}
