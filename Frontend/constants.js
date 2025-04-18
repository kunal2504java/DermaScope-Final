export const API_BASE_URL = 'http://localhost:8081';

export const AUTH_ROUTES = {
  SIGNUP: '/signup',
  LOGIN: '/login',
  CHECK: '/authenticate',
  LOGOUT: '/logout'
};

export const APP_ROUTES = {
  HOME: '/',
  SIGNUP: '/signup',
  LOGIN: '/login',
  TAKE_TEST: '/take-test',
  TEST_RESULTS: '/test-results'
};

export const API_ROUTES = {
  LOGIN: `${API_BASE_URL}/login`,
  SIGNUP: `${API_BASE_URL}/signup`,
  LOGOUT: `${API_BASE_URL}/logout`,
  AUTHENTICATE: `${API_BASE_URL}/authenticate`,
  PREDICT: `${API_BASE_URL}/predict`,
  SHARE_RESULTS: `${API_BASE_URL}/share-results`
};