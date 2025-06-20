CREATE TABLE simple.create_table_as_select_from_nested_cte_with_two_join_union_coalesce AS
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

UNION

SELECT COALESCE(user_history.user_id, imported_users.user_id) AS user_id,
       COALESCE(user_history.name, imported_users.name) AS name,
       COALESCE(user_history.status, imported_users.status) AS status,
       COALESCE(user_history.project_id, imported_users.project_id) AS project_id
  FROM source.user_history

       FULL OUTER JOIN source.imported_users
       ON imported_users.user_id = user_history.user_id
          AND imported_users.project_id = user_history.project_id
; 
