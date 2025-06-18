CREATE TABLE simple.create_table_as_select_from_nested_cte_with_join AS
WITH _cte AS (
      WITH _nested_cte AS (
              SELECT users.id AS user_id,
                     users.name
                FROM source.users
      )
      SELECT _nested_cte.*,
             user_status.status
        FROM _nested_cte

             LEFT JOIN source.user_status
             ON user_status.user_id = _nested_cte.user_id
)
SELECT _cte.user_id,
       _cte.name,
       _cte.status
  FROM _cte
; 
