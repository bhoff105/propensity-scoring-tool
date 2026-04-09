-- suppression.sql
-- Builds the customer exclusion list for email/digital campaigns.
-- Any customer in this table should be suppressed from outbound targeting.
--
-- Exclusion criteria fall into four categories:
--   1. Invalid or undeliverable addresses
--   2. Internal company employees (identified by corporate email domains)
--   3. Addresses with unusually high customer density (likely commercial/shared addresses)
--   4. Deceased customers
--
-- This suppression list is applied as an exclusion join in downstream
-- audience-building queries (see reactivation.sql for an example).


CREATE OR REPLACE TABLE customer_data.dm_suppression AS (

  WITH

  -- ── Category 1: Invalid or undeliverable addresses ──────────────────────
  -- Covers: USPS nixie flag, non-deliverable codes, non-US addresses,
  -- military/territory addresses, missing name/address fields,
  -- and known corporate headquarters addresses.
  invalid_addresses AS (
    SELECT *
    FROM customer_data.vw_customer
    WHERE
      nixie_flag = 'X'                             -- USPS-flagged undeliverable
      OR mailable_flag != 'Y'
      OR address_deliverability_code IN ('C', 'H') -- commercial or high-rise unverified
      OR state_abbrev IN (
        'AE', 'AP', 'AS', 'VI', 'PR', 'GU', 'HI'  -- APO/FPO, territories, Hawaii
      )
      OR country != 'USA'
      OR city IN ('APO', 'FPO', 'DPO')             -- military mail codes
      -- Corporate HQ addresses (internal fulfillment locations, not consumers)
      OR address ILIKE '%412 BRIARWOOD WAY%'
      OR address ILIKE '%88 RIDGELINE DR%'
      -- Missing required fields
      OR first_name IS NULL OR TRIM(first_name) IN ('', '.')
      OR last_name  IS NULL OR TRIM(last_name)  IN ('', '.')
      OR address    IS NULL OR TRIM(address)    IN ('', '.')
      OR city       IS NULL OR TRIM(city)       IN ('', '.')
      OR state_abbrev IS NULL
      OR zip5 IS NULL
  ),

  -- ── Category 2: Internal employees ──────────────────────────────────────
  -- Identifies customers whose orders are associated with corporate
  -- email domains. These should never receive consumer campaigns.
  internal_employees AS (
    SELECT DISTINCT enterprise_id
    FROM customer_data.vw_order_header
    WHERE
      email_address ILIKE '%@briarwoodgoods.com%'
      OR email_address ILIKE '%@briarwoodfulfillment.com%'
      -- Add additional internal domains here as needed
  ),

  -- ── Category 3: High-density addresses ──────────────────────────────────
  -- More than 3 distinct customers at the same address likely indicates
  -- a commercial address, mail forwarding service, or data quality issue.
  address_counts AS (
    SELECT address, COUNT(DISTINCT enterprise_id) AS customer_count
    FROM customer_data.vw_customer
    GROUP BY address
    HAVING customer_count > 3
  ),
  high_density_addresses AS (
    SELECT DISTINCT enterprise_id
    FROM customer_data.vw_customer
    WHERE address IN (SELECT address FROM address_counts)
  ),

  -- ── Category 4: Deceased ─────────────────────────────────────────────────
  deceased AS (
    SELECT enterprise_id
    FROM customer_data.vw_customer
    WHERE deceased_flag = 'Y'
  ),

  -- ── Union all exclusion sources ──────────────────────────────────────────
  all_suppressed AS (
    SELECT DISTINCT enterprise_id FROM invalid_addresses
    UNION
    SELECT DISTINCT enterprise_id FROM internal_employees
    UNION
    SELECT DISTINCT enterprise_id FROM high_density_addresses
    UNION
    SELECT DISTINCT enterprise_id FROM deceased
  )

  SELECT * FROM all_suppressed

);
