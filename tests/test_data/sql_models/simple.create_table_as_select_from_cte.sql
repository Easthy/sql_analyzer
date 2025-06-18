CREATE TABLE simple.create_table_as_select_from_cte AS
WITH _cte AS (
       SELECT users.id AS user_id,
              users.name
         FROM source.users
)
SELECT _cte.user_id,
       _cte.name
  FROM _cte
;