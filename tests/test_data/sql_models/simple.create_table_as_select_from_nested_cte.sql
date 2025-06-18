CREATE TABLE simple.create_table_as_select_from_nested_cte AS
WITH _cte AS (
      WITH _nested_cte AS (
              SELECT users.id AS user_id,
                     users.name
                FROM source.users
      )
      SELECT _nested_cte.*
        FROM _nested_cte
)
SELECT _cte.user_id,
       _cte.name
  FROM _cte
;