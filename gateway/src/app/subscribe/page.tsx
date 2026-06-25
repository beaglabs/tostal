import { redirect } from "next/navigation";
import { stripe } from "@/lib/stripe";

const PRICE_ID = process.env.STRIPE_PRICE_ID || "price_1TlexE8k0ubC0hdJrGUVKQI9";

export default async function SubscribePage() {
  const session = await stripe.checkout.sessions.create({
    mode: "subscription",
    line_items: [{ price: PRICE_ID, quantity: 1 }],
    success_url: `${process.env.BETTER_AUTH_URL}/app`,
    cancel_url: `${process.env.BETTER_AUTH_URL}/subscribe`,
  });

  if (session.url) {
    redirect(session.url);
  }

  return <p>Redirecting to checkout...</p>;
}