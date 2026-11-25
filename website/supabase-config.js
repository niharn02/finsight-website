// Shared Supabase client setup for FinSight's static Cloudflare Pages website.
const SUPABASE_URL = "https://iglfciogemwqpfhqhwkl.supabase.co";
const SUPABASE_PUBLISHABLE_KEY = "sb_publishable_Bhz5fQTPgDVLG2MToojskg_s2gZkug3";
// Allow only known dashboard origins passed from the dashboard auth gate.
const ALLOWED_DASHBOARD_ORIGINS = [
  "https://finsight-dashboard-xtk6.onrender.com",
  "https://finsight-dashboard-pr-3.onrender.com"
];
const ret = new URLSearchParams(location.search).get("return");
const FINSIGHT_RENDER_URL = ALLOWED_DASHBOARD_ORIGINS.includes(ret) ? ret : ALLOWED_DASHBOARD_ORIGINS[0];
window.supabaseClient = supabase.createClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);

// Exchange a Supabase token for a short-lived, one-time dashboard handoff.
async function finsightHandoff(session) {
  if (!session?.access_token) throw new Error("No active authentication session.");
  const response = await fetch(FINSIGHT_RENDER_URL + "/auth/handoff/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ access_token: session.access_token })
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || !payload.redirect_url) throw new Error(payload.error || "Dashboard sign-in handoff failed.");
  location.assign(payload.redirect_url);
}
document.addEventListener("DOMContentLoaded", async () => {
  const client = window.supabaseClient, status = document.getElementById("auth-message");
  const message = (text, error=false) => { if(status){status.textContent=text;status.className="mt-3 text-sm "+(error?"text-red-600":"text-primary");} };
  document.querySelectorAll("[data-google-auth]").forEach(button => button.addEventListener("click", async () => {
    button.disabled=true;message("Redirecting to Google...");
    const {error}=await client.auth.signInWithOAuth({provider:"google",options:{redirectTo:location.origin+"/sign_in.html?handoff=1"}});
    if(error){message(error.message,true);button.disabled=false;}
  }));
  const {data:{session}}=await client.auth.getSession();
  if(session&&(new URLSearchParams(location.search).has("handoff")||location.hash.includes("access_token"))){message("Signed in. Opening dashboard...");try{await finsightHandoff(session);}catch(e){message(e.message,true);}}
  const form=document.getElementById("signin-form");
  if(form)form.addEventListener("submit",async event=>{event.preventDefault();const btn=form.querySelector('[type="submit"]');btn.disabled=true;message("Signing in...");
    const {data,error}=await client.auth.signInWithPassword({email:document.getElementById("email").value.trim(),password:document.getElementById("password").value});
    if(error){message(error.message,true);btn.disabled=false;return;}try{await finsightHandoff(data.session);}catch(e){message(e.message,true);btn.disabled=false;}
  });
  const signup=document.getElementById("signup-form");
  if(signup)signup.addEventListener("submit",async event=>{event.preventDefault();const btn=signup.querySelector('[type="submit"]');btn.disabled=true;const password=document.getElementById("password").value,confirm=document.getElementById("confirm-password").value;if(password!==confirm){message("Passwords do not match.",true);btn.disabled=false;return;}message("Creating your account...");
    const {data,error}=await client.auth.signUp({email:document.getElementById("email").value.trim(),password,options:{emailRedirectTo:location.origin+"/sign_in.html?handoff=1"}});
    if(error){message(error.message,true);btn.disabled=false;return;}if(!data.session){message("Account created. Check your email to confirm, then sign in.");btn.disabled=false;return;}try{await finsightHandoff(data.session);}catch(e){message(e.message,true);btn.disabled=false;}
  });
  const reset=document.getElementById("forgot-password");if(reset)reset.addEventListener("click",async()=>{const email=document.getElementById("email").value.trim();if(!email){message("Enter your email address first.",true);return;}const {error}=await client.auth.resetPasswordForEmail(email,{redirectTo:location.origin+"/sign_in.html"});message(error?error.message:"Password reset email sent.",!!error);});
});