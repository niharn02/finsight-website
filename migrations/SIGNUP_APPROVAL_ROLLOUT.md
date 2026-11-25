# FinSight signup approval rollout (feature branch)

This feature is deliberately not deployed or merged. Follow this checklist only when ready to roll it out.

## 1. Database migration
Run `migrations/signup_approval.sql` in the Supabase SQL editor for the FinSight project.
It creates request, audit and notification tables and a trigger for new Auth accounts.
It backfills existing Auth users as approved so existing users retain access.

## 2. Admin bootstrap
In Supabase Auth, copy the UUID for the account that will administer access.
On the Render backend service, configure `FINANCE_ADMIN_USER_IDS` as a comma-separated list of trusted UUIDs.
Never put this value in static frontend files or permit users to self-assign admin roles.
Restart/redeploy backend only when separately approved.

## 3. Access-management page
After rollout, visit `/admin_access.html` on the Cloudflare Pages site and sign in with the admin identity.
The page calls the protected admin API; non-admins receive HTTP 403.
Admin actions approve, reject, or revoke. Each action records an audit entry and creates an in-app notification.

## 4. API enforcement
The auth handoff rejects non-approved accounts. The `@require_supabase_auth` decorator also checks approval status for protected API routes.
Admin UUIDs are treated as approved.
If the database cannot be checked, access fails closed with a service error.

## 5. Test checklist before rollout
- Existing users: remain approved after migration and can access dashboard.
- New Google signup and email signup: pending; handoff is blocked.
- Admin: can list requests and approve/reject/revoke.
- Approved account: dashboard API calls work.
- Pending/rejected/revoked accounts: protected API access is denied.
- Non-admin calls to admin endpoints: 403.
- Audit records and notification entries are created for admin decisions.
- Browser CORS/preflight works from the Cloudflare Pages origin.
- Verify the migration against a backup and test environment before production.

## Limitations
Email notifications are not configured in this feature because no email service/provider secrets or sender settings were available. In-app notification records are created on administrative status changes. Notification retrieval is available through `GET /api/access/notifications`.
