BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS simple.create_table_cte_join_aggregate_on_window (
    user_id INTEGER,
    email VARCHAR,
    status VARCHAR,
    payment_cnt INTEGER ENCODE AZ64
)
    DISTSTYLE KEY
    DISTKEY(user_id)
    SORTKEY(status, user_id)
;

INSERT INTO simple.create_table_cte_join_aggregate_on_window
WITH _cte AS (
    SELECT user.id,
           user.email,
           ROW_NUMBER() OVER (
                  PARTITION BY user.id
                      ORDER BY user.load_date DESC
           ) AS rn
      FROM source.user
)
  SELECT _cte.id,
         _cte.email,
         user_status.status,
         COUNT(
             CASE
                  WHEN user_payment.operation_id IN (1, 2)
                       THEN user_payment.id
                  ELSE NULL
             END
         ) AS payment_cnt
    FROM _cte
  
         LEFT JOIN source.user_status
         ON user_status.user_id = _cte.id
  
         LEFT JOIN source.user_payment
         ON user_payment.user_id = _cte.id

   WHERE _cte.rn = 1
GROUP BY 1, 2
;

END TRANSACTION;