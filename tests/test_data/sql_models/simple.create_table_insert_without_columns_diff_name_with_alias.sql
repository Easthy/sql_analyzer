BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS simple.create_table_insert_without_columns_diff_name_with_alias (
    user_id INTEGER,
    registration_date DATE ENCODE DELTA
)
    DISTSTYLE KEY
    DISTKEY(user_id)
    SORTKEY(registration_date)
;

INSERT INTO simple.create_table_insert_without_columns_diff_name_with_alias
SELECT mysql_users.user_id,
       mysql_users.reg_date AS registration_date
  FROM mwl._mysql_users
;

END TRANSACTION;