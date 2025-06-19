CREATE TABLE simple.simple.create_table_as_select_from_nested_cte_with_two_join AS
WITH _cte AS (
      WITH _nested_cte AS (
              SELECT users.id AS user_id,
                     users.name,
                     project.project_id
                FROM source.users

                     INNER JOIN reference.project
                     ON users.project_id = users.project_id
      )
      SELECT _nested_cte.*,
             user_status.status
        FROM _nested_cte

             LEFT JOIN source.user_status
             ON user_status.user_id = _nested_cte.user_id
)
SELECT _cte.user_id,
       _cte.name,
       _cte.status,
       _cte.project_id
  FROM _cte
; 
