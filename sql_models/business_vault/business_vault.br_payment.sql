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
       sum_usd,
       in_out_usd
)
SELECT mysql_payments.user_hash_key,
       users.fake,
       mysql_payments.project_id,
       mysql_payments.closed_at,
       mysql_payments.operation_id,
       mysql_payments.status,
       mysql_payments.sum,
       mysql_payments.sum_usd / 100.0 AS sum_usd,
       SUM(
              CASE
                     WHEN mysql_payments.operation_id = 11
                          THEN 1
                     ELSE -1
              END * mysql_payments.sum_usd / 100.0
       ) OVER (
              PARTITION BY mysql_payments.user_hash_key
                  ORDER BY mysql_payments.closed_at DESC
       ) AS in_out_usd
  FROM mwl.mysql_payments

       LEFT JOIN (
              SELECT users.user_hash_key,
                     users.fake
                FROM mwl.users
               WHERE users.project_id NOT IN (4, 5, 6)
       ) AS users
 WHERE mysql_payments.closed_at >= CURRENT_DATE - 30
;

END TRANSACTION;