BEGIN TRANSACTION;

DELETE
  FROM business_vault.br_payment
 WHERE br_payment.closed_at >= CURRENT_DATE - 30
;

INSERT INTO business_vault.br_payment(
       user_hash_key,
       fake,
       project_id,
       closed_at,
       operation_id,
       status,
       sum,
       sum_usd
)
SELECT mysq_payments.user_hash_key,
       users.fake,
       mysq_payments.project_id,
       mysq_payments.closed_at,
       mysq_payments.operation_id,
       mysq_payments.status,
       mysq_payments.sum,
       mysq_payments.sum_usd / 100.0 AS sum_usd
  FROM mwl.mysq_payments

       LEFT JOIN (
              SELECT users.user_hash_key,
                     users.fake
                FROM mwl.users
               WHERE users.project_id NOT IN (4, 5, 6)
       ) AS users
 WHERE mysq_payments.closed_at >= CURRENT_DATE - 30
;

END TRANSACTION;