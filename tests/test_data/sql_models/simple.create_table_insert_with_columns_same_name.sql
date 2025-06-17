BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS simple.create_insert_with_columns_same_name (
    user_id INTEGER,
    registration_date DATE ENCODE DELTA
)
    DISTSTYLE KEY
    DISTKEY(user_id)
    SORTKEY(registration_date)
;

INSERT INTO simple.create_insert_with_columns_same_name (
    user_id,
    registration_date
)
SELECT mysql_users.user_id,
       mysql_users.registration_date
  FROM mwl.mysql_users
;

END TRANSACTION;