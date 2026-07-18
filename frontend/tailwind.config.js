/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0b0f19",
        card: "rgba(17, 24, 39, 0.6)",
        primary: {
          light: "#38bdf8",
          DEFAULT: "#0284c7",
          dark: "#0369a1",
        },
        secondary: {
          light: "#818cf8",
          DEFAULT: "#4f46e5",
          dark: "#3730a3",
        },
        slate: {
          750: "#1e293b",
        }
      },
    },
  },
  plugins: [],
}
