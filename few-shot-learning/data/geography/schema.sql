CREATE TABLE border_info (state_name text,border text);
CREATE TABLE city (city_name text,population int(11),country_name varchar(3),state_name text);
CREATE TABLE highlow (state_name text,highest_elevation text,lowest_point text,highest_point text,lowest_elevation text);
CREATE TABLE lake (lake_name text,area double,country_name varchar(3),state_name text);
CREATE TABLE mountain (mountain_name text,mountain_altitude int(11),country_name varchar(3),state_name text);
CREATE TABLE river (river_name text,length int(11),country_name varchar(3),traverse text);
CREATE TABLE state (state_name text,population int(11),area double,country_name varchar(3),capital text,density double);