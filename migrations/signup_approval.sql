-- FinSight signup approval migration.
-- Run in Supabase SQL Editor. Backfill existing accounts as approved BEFORE enabling
-- backend enforcement; obtain UUIDs from Supabase Auth, never from browser input.
CREATE TABLE IF NOT EXISTS public.finsight_access_requests (
    user_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email text,
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected','revoked')),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS public.finsight_access_notifications (
    notification_id bigserial PRIMARY KEY,
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    message text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    read_at timestamptz
);
CREATE TABLE IF NOT EXISTS public.finsight_access_audit (
    audit_id bigserial PRIMARY KEY,
    actor_user_id uuid NOT NULL REFERENCES auth.users(id),
    target_user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    action text NOT NULL,
    old_status text,
    new_status text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.finsight_access_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.finsight_access_notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.finsight_access_audit ENABLE ROW LEVEL SECURITY;
-- These tables are accessed only by the backend using DATABASE_URL. No client policies are granted.
REVOKE ALL ON public.finsight_access_requests, public.finsight_access_notifications, public.finsight_access_audit FROM anon, authenticated;
