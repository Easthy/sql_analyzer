BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS simple.create_table (
       user_id INTEGER,
       registration_date DATE ENCODE DELTA
)
       DISTSTYLE KEY
       DISTKEY(user_id)
       SORTKEY(registration_date)
;

END TRANSACTION;