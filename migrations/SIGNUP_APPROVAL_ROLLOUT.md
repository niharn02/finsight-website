# FinSight signup approval rollout (feature branch)

This feature is not deployed or merged. Test only in a disposable Supabase project first.

## 1. Test-environment preparation
- Use a separate Supabase project and test backend environment. Never use the production database for integration tests.
- Confirm the backend test environment has its own `DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `HANDOFF_ENCRYPTION_KEY`.
- Back up the test database before running the migration.
- Do not configure production admin IDs or deploy production changes during this test.

## 2. Apply and verify the migration in test
Run `migrations/signup_approval.sql` in the test SQL editor. It creates access-request, notification and audit tables, backfills current Auth users as approved (without overwriting existing rows), and installs the signup trigger.

After migration, verify:

```sql
-- Existing Auth users should have an approval row.
SELECT COUNT(*) AS auth_users FROM auth.users;
SELECT COUNT(*) AS approval_rows FROM public.finsight_access_requests;

-- Inspect status distribution.
SELECT status, COUNT(*) FROM public.finsight_access_requests GROUP BY status;

-- Inspect trigger installation.
SELECT tgname, tgenabled
FROM pg_trigger
WHERE tgrelid = 'auth.users'::regclass
  AND tgname = 'finsight_access_request_on_signup'
  AND NOT tgisinternal;

-- Confirm the function definition and security attributes.
SELECT p.proname, p.prosecdef, p.proconfig
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'public'
  AND p.proname = 'finsight_create_access_request';
```

Test a new signup and verify a pending row is created automatically. Do not manually alter production Auth users.

## 3. Admin bootstrap (test environment)
Configure `FINANCE_ADMIN_USER_IDS` with the UUID of the test administrator only on the test backend. Keep the UUID server-side. Do not self-assign admin status from frontend metadata.

## 4. Deploy and test the feature in isolation
Deploy the feature branch only to the test backend and test static site preview. Confirm the test backend is connected exclusively to the test Supabase project.

### Test matrix
- **Existing user:** backfilled approved; protected APIs work.
- **New Google signup:** pending; dashboard handoff denied; protected API denied.
- **New email/password signup:** pending; same enforcement as Google.
- **Approval:** pending account becomes approved; handoff and protected APIs succeed.
- **Reject:** rejected account cannot handoff or use protected APIs.
- **Revoke:** previously approved account is denied on protected APIs and subsequent handoff.
- **Revocation during handoff:** initiate a handoff, revoke access before code exchange, then verify exchange is denied.
- **Admin authorization:** admin can list and action requests; non-admin receives 403.
- **Input validation:** malformed UUID and unsupported action are rejected; unknown target returns 404.
- **Audit/notification:** each successful decision records audit and notification; failed DB transaction creates neither partial state nor a successful response.
- **Atomic handoff:** code is single-use, expires, and is not usable from an unapproved account.
- **Origin controls:** unexpected origins are rejected; expected origins pass preflight.
- **DB outage:** approval checks fail closed (503); no protected API or handoff succeeds.
- **User isolation:** test with two approved users and confirm neither can access the other's data or notifications.

Record actual test results and logs (without access tokens, passwords, or sensitive financial data) before deciding to roll out.

## 5. Production rollout (requires explicit approval)
1. Back up the production database and confirm restore procedure.
2. Confirm the production administrator UUID and set `FINANCE_ADMIN_USER_IDS` on the production backend.
3. Schedule the rollout: the backend enforcement code fails closed if the approval table is absent, so deploying it before the migration can cause temporary sign-in/API downtime. The trigger also makes new signups pending as soon as the migration is applied. Coordinate the two steps closely.
4. Deploy enforcement and apply the migration in the agreed sequence, immediately verify admin and existing-user access, and monitor errors. Do not continue if either step fails.
5. Keep a rollback plan. Rollback must not simply remove enforcement while the signup trigger continues to create pending rows; verify the intended state first.

## 6. Access management
The admin page is `/admin_access.html`. Admin actions approve, reject or revoke. Successful actions record audit history and in-app notifications. Notifications are read through `GET /api/access/notifications`.

## Limitations
Email notifications are not configured. The feature currently creates in-app notification records only.
