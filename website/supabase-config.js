// Shared Supabase client setup for the Render-hosted FinSight website.
// Production uses the existing production Supabase project. PR previews/local testing
// use the isolated test project so preview signups never affect production accounts.
const PRODUCTION_HOST = "finsight-dashboard-xtk6.onrender.com";
const PRODUCTION_SUPABASE_URL = "https://iglfciogemwqpfhqhwkl.supabase.co";
const PRODUCTION_PUBLISHABLE_KEY = "sb_publishable_Bhz5fQTPgDVLG2MToojskg_s2gZkug3";
const TEST_SUPABASE_URL = "https://lzhmdjqvjantmkmlgbej.supabase.co";
const TEST_PUBLISHABLE_KEY = "sb_publishable_Tf0pJqN4idxzkUXvUy_1bw_WqVjkuSp";

const isProductionHost = location.hostname === PRODUCTION_HOST;
const SUPABASE_URL = isProductionHost ? PRODUCTION_SUPABASE_URL : TEST_SUPABASE_URL;
const SUPABASE_PUBLISHABLE_KEY = isProductionHost ? PRODUCTION_PUBLISHABLE_KEY : TEST_PUBLISHABLE_KEY;

// Render website and dashboard share an origin. Local standalone testing is retained.
const ALLOWED_DASHBOARD_ORIGINS = new Set([
  location.origin,
  "https://finsight-dashboard-xtk6.onrender.com",
  "http://localhost:5000",
  "http://127.0.0.1:5000"
]);
const DEFAULT_DASHBOARD_ORIGIN = location.origin;

window.supabaseClient = supabase.createClient(
  SUPABASE_URL,
  SUPABASE_PUBLISHABLE_KEY
);

// Accept only allowlisted local dashboard destinations.
function finsightReturnOrigin() {
  const candidate = new URLSearchParams(location.search).get("return");
  return ALLOWED_DASHBOARD_ORIGINS.has(candidate)
    ? candidate
    : DEFAULT_DASHBOARD_ORIGIN;
}

// Exchange Supabase session for a one-time dashboard handoff.
async function finsightHandoff(session) {
  if (!session?.access_token) {
    throw new Error("No active authentication session.");
  }

  const targetOrigin = finsightReturnOrigin();

  const response = await fetch(targetOrigin + "/auth/handoff/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      access_token: session.access_token,
      return_origin: targetOrigin
    })
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok || !payload.redirect_url) {
    const accessError = new Error(
      payload.error || "Dashboard sign-in handoff failed."
    );
    accessError.accessStatus = payload.access_status || null;
    throw accessError;
  }

  location.assign(payload.redirect_url);
}

document.addEventListener("DOMContentLoaded", async () => {
  const client = window.supabaseClient;
  const status = document.getElementById("auth-message");

  const message = (text, error = false) => {
    if (status) {
      status.textContent = text;
      status.className =
        "mt-3 text-sm " + (error ? "text-red-600" : "text-primary");
    }
  };

  // OAuth callback URL for the locally hosted frontend.
  const callbackUrl = () =>
    location.origin +
    "/sign_in.html?handoff=1&return=" +
    encodeURIComponent(finsightReturnOrigin());

  // Google OAuth.
  document.querySelectorAll("[data-google-auth]").forEach(button =>
    button.addEventListener("click", async () => {
      button.disabled = true;
      message("Redirecting to Google...");

      const { error } = await client.auth.signInWithOAuth({
        provider: "google",
        options: {
          redirectTo: callbackUrl()
        }
      });

      if (error) {
        message(error.message, true);
        button.disabled = false;
      }
    })
  );

  // Retrieve the existing authentication session.
  const {
    data: { session }
  } = await client.auth.getSession();

  // Complete OAuth or token-based handoff.
  if (
    session &&
    (
      new URLSearchParams(location.search).has("handoff") ||
      location.hash.includes("access_token")
    )
  ) {
    message("Signed in. Opening dashboard...");

    try {
      await finsightHandoff(session);
    } catch (e) {
      message(
        e.accessStatus === "pending"
          ? "Your FinSight account is awaiting administrator approval. You can sign in again after approval."
          : e.accessStatus === "rejected"
          ? "Your access request was rejected. Please contact the administrator."
          : e.accessStatus === "revoked"
          ? "Your FinSight access has been revoked. Please contact the administrator."
          : e.message,
        true
      );
    }
  }

  // Email/password sign-in.
  const form = document.getElementById("signin-form");

  if (form) {
    form.addEventListener("submit", async event => {
      event.preventDefault();

      const btn = form.querySelector('[type="submit"]');
      btn.disabled = true;
      message("Signing in...");

      const { data, error } = await client.auth.signInWithPassword({
        email: document.getElementById("email").value.trim(),
        password: document.getElementById("password").value
      });

      if (error) {
        message(error.message, true);
        btn.disabled = false;
        return;
      }

      try {
        await finsightHandoff(data.session);
      } catch (e) {
        message(e.message, true);
        btn.disabled = false;
      }
    });
  }

  // Email/password signup.
  const signup = document.getElementById("signup-form");

  if (signup) {
    signup.addEventListener("submit", async event => {
      event.preventDefault();

      const btn = signup.querySelector('[type="submit"]');
      btn.disabled = true;

      const password = document.getElementById("password").value;
      const confirm = document.getElementById("confirm-password").value;

      if (password !== confirm) {
        message("Passwords do not match.", true);
        btn.disabled = false;
        return;
      }

      message("Creating your account...");

      const { data, error } = await client.auth.signUp({
        email: document.getElementById("email").value.trim(),
        password,
        options: {
          emailRedirectTo: callbackUrl()
        }
      });

      if (error) {
        message(error.message, true);
        btn.disabled = false;
        return;
      }

      // Email verification required.
      if (!data.session) {
        message("Account created. Check your email to confirm, then sign in.");
        btn.disabled = false;
        return;
      }

      // Immediate handoff if signup returns a session.
      try {
        await finsightHandoff(data.session);
      } catch (e) {
        message(e.message, true);
        btn.disabled = false;
      }
    });
  }

  // Password reset.
  const reset = document.getElementById("forgot-password");

  if (reset) {
    reset.addEventListener("click", async () => {
      const email = document.getElementById("email").value.trim();

      if (!email) {
        message("Enter your email address first.", true);
        return;
      }

      const { error } = await client.auth.resetPasswordForEmail(email, {
        redirectTo: location.origin + "/sign_in.html"
      });

      message(
        error ? error.message : "Password reset email sent.",
        !!error
      );
    });
  }
});
