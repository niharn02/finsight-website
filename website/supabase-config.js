// Shared Supabase client setup for FinSight's static Cloudflare Pages website.
const SUPABASE_URL = "https://iglfciogemwqpfhqhwkl.supabase.co";
const SUPABASE_PUBLISHABLE_KEY = "sb_publishable_Bhz5fQTPgDVLG2MToojskg_s2gZkug3";

// Only these exact dashboard origins are allowed as handoff destinations.
const ALLOWED_DASHBOARD_ORIGINS = new Set([
  "https://finsight-dashboard-xtk6.onrender.com",
  "https://finsight-dashboard-pr-3.onrender.com"
]);

// The production dashboard is the default destination when no valid return origin is supplied.
const DEFAULT_DASHBOARD_ORIGIN = "https://finsight-dashboard-xtk6.onrender.com";

window.supabaseClient = supabase.createClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);

// Accept a return destination only when it exactly matches the allowlisted dashboard origins.
function finsightReturnOrigin() {
  const candidate = new URLSearchParams(location.search).get("return");
  return ALLOWED_DASHBOARD_ORIGINS.has(candidate)
    ? candidate
    : DEFAULT_DASHBOARD_ORIGIN;
}

// Exchange a Supabase token for a short-lived, one-time dashboard handoff.
async function finsightHandoff(session) {
  if (!session?.access_token) {
    throw new Error("No active authentication session.");
  }

  // Use the validated destination for both the handoff endpoint and return-origin payload.
  const targetOrigin = finsightReturnOrigin();

  const response = await fetch(targetOrigin + "/auth/handoff/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },

    // The backend independently validates this origin against its own allowlist.
    body: JSON.stringify({
      access_token: session.access_token,
      return_origin: targetOrigin
    })
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok || !payload.redirect_url) {
    const accessError = new Error(payload.error || "Dashboard sign-in handoff failed."); accessError.accessStatus = payload.access_status || null; throw accessError;
  }

  // Redirect only after the backend successfully generates the handoff URL.
  location.assign(payload.redirect_url);
}

document.addEventListener("DOMContentLoaded", async () => {
  const client = window.supabaseClient;
  const status = document.getElementById("auth-message");

  // Update the authentication status message displayed on the page.
  const message = (text, error = false) => {
    if (status) {
      status.textContent = text;
      status.className =
        "mt-3 text-sm " + (error ? "text-red-600" : "text-primary");
    }
  };

  // Preserve the validated dashboard destination across Google's OAuth round-trip.
  const callbackUrl = () =>
    location.origin +
    "/sign_in.html?handoff=1&return=" +
    encodeURIComponent(finsightReturnOrigin());

  // Handle Google OAuth sign-in buttons.
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

  // Complete dashboard handoff when returning from OAuth or when a token is present.
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
      message(e.accessStatus === "pending" ? "Your FinSight account is awaiting administrator approval. You can sign in again after approval." : e.accessStatus === "rejected" ? "Your access request was rejected. Please contact the administrator." : e.accessStatus === "revoked" ? "Your FinSight access has been revoked. Please contact the administrator." : e.message, true);
    }
  }

  // Handle email/password sign-in.
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

  // Handle account creation with email/password.
  const signup = document.getElementById("signup-form");

  if (signup) {
    signup.addEventListener("submit", async event => {
      event.preventDefault();

      const btn = signup.querySelector('[type="submit"]');
      btn.disabled = true;

      const password = document.getElementById("password").value;
      const confirm = document.getElementById("confirm-password").value;

      // Ensure the user has entered the same password in both fields.
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

      // If email confirmation is required, ask the user to verify their email.
      if (!data.session) {
        message("Account created. Check your email to confirm, then sign in.");
        btn.disabled = false;
        return;
      }

      // Immediately hand off the session if signup returned an active session.
      try {
        await finsightHandoff(data.session);
      } catch (e) {
        message(e.message, true);
        btn.disabled = false;
      }
    });
  }

  // Handle password-reset requests.
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