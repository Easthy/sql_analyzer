BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS simple.create_table_as_select_without_from AS
SELECT 1 AS user_id,
       CURRENT_DATE AS registration_date
;

END TRANSACTION;