BEGIN TRANSACTION;

DELETE 
  FROM public.user_daily_kpi
 WHERE user_daily_kpi.date >= CURRENT_DATE - 30
;

INSERT INTO public.user_daily_kpi (
    user_hash_key,
    project_id,
    date,
    deposit_usd,
    withdrawal_usd
)
  SELECT br_payment.user_hash_key,
         br_payment.project_id,
         br_payment.closed_at::date AS date,
         SUM(
              CASE
                  WHEN br_payment.operation_id = 1 
                       THEN br_payment.sum_usd
              END
         ) AS deposit_usd,
         SUM(
              CASE
                  WHEN br_payment.operation_id = 2 
                       THEN br_payment.sum_usd
              END
         ) AS withdrawal_usd
    FROM business_vault.br_payment
   WHERE br_payment.status = 3
         AND br_payment.fake IS NOT TRUE
GROUP BY br_payment.user_hash_key,
         br_payment.project_id,
         br_payment.closed_at::date
;

END TRANSACTION;