CREATE TABLE simple.simple.create_table_as_select_join_case AS
SELECT users.id AS user_id,
       users.name,
       CASE
            WHEN users.id IS NOT NULL
                 THEN users.status
            ELSE user_history.status
       END AS status
  FROM source.users

       LEFT JOIN source.user_history
       ON user_history.user_id = users.id
;