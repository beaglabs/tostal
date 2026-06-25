"use client";

import { useState } from "react";
import { authClient } from "@/lib/auth-client";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  return (
    <main>
      <h1>Sign in to Tostal</h1>
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          await authClient.signIn.email({ email, password });
        }}
      >
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <button type="submit">Sign in</button>
      </form>
      <button onClick={() => authClient.signIn.social({ provider: "google" })}>
        Sign in with Google
      </button>
      <button onClick={() => authClient.signIn.social({ provider: "github" })}>
        Sign in with GitHub
      </button>
    </main>
  );
}