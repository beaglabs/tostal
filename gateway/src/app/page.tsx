import Link from "next/link";

export default function HomePage() {
  return (
    <main>
      <h1>Tostal Sci-data Platform</h1>
      <p>Cloud-hosted geoscience data platform with Jupyter notebook frontend.</p>
      <nav>
        <Link href="/login">Sign in</Link>
      </nav>
    </main>
  );
}