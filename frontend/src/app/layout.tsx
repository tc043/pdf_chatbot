import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Enterprise PDF RAG Chatbot",
  description: "A production-grade PDF retrieval and conversation system.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased bg-[#0b0f19] text-[#e2e8f0]">
        {children}
      </body>
    </html>
  );
}
