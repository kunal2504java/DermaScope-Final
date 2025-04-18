import axios from "axios";
import {API_BASE_URL, AUTH_ROUTES, APP_ROUTES, API_ROUTES} from "./constants.js";

export function validateEmail(email) {
    const re = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return re.test(String(email).toLowerCase());
}

export async function authUserHomeLoader() {
    try {
        const response = await axios.get(API_BASE_URL + AUTH_ROUTES.CHECK, {withCredentials: true});

        if (response.status >= 200 && response.status < 300)
            return response.data;
    }catch (error) {

    }
    return null;
}

export async function testResultsLoader() {
    try {
        // Add a timestamp to prevent caching
        const timestamp = new Date().getTime();
        const url = `${API_BASE_URL}${API_ROUTES.PREDICT}?t=${timestamp}`;
        
        const response = await axios.get(url, {
            headers: {
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            },
            withCredentials: true
        });

        if (response.status >= 200 && response.status < 300)
            return response.data;
    }catch (error) {

    }
    return null;
}

export async function sharedResultsLoader({params}) {
    try {
        // Add a timestamp to prevent caching
        const timestamp = new Date().getTime();
        const url = `${API_BASE_URL}${API_ROUTES.SHARE_RESULTS}/${params.key}?t=${timestamp}`;
        
        const response = await axios.get(url, {
            headers: {
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            },
            withCredentials: true
        });

        if (response.status >= 200 && response.status < 300)
            return response.data;
    }catch (error) {

    }
    return null;
}

export const getInitials = (name) => {
    if (!name) return '';
    const parts = name.split(' ')
    let initials = '';
    for (let i = 0; i < parts.length && i < 2; i++) {
        if (parts[i].length > 0 && parts[i] !== '') {
            initials += parts[i][0]
        }
    }
    return initials
}